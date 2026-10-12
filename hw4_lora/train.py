"""Full FT и однофакторное сравнение LoRA на SST-2."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

MODEL='Qwen/Qwen2.5-1.5B-Instruct'
DATASET='stanfordnlp/sst2'
SEED=42
PROMPT='Classify the sentiment of the movie review. Reply with exactly one word: positive or negative.'
BASE=dict(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['q_proj','v_proj'])
CONFIGS={'base':None,'full':None,'lora':BASE,
 'rank32':{**BASE,'r':32},'alpha64':{**BASE,'lora_alpha':64},
 'dropout0':{**BASE,'lora_dropout':0.0},
 'all_linear':{**BASE,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}}

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def prepare():
    out=Path('data');out.mkdir(exist_ok=True)
    if Path('data_manifest.json').exists():
        meta=json.loads(Path('data_manifest.json').read_text())
    else:
        api=HfApi();meta={'model':MODEL,'model_revision':api.model_info(MODEL).sha,
            'dataset':DATASET,'dataset_revision':api.dataset_info(DATASET).sha,'seed':SEED}
    ds=load_dataset(DATASET,revision=meta['dataset_revision'])
    test=[dict(id=int(x['idx']),text=x['sentence'],label=int(x['label'])) for x in ds['validation']]
    normalize=lambda s:' '.join(s.lower().split())
    heldout={normalize(x['text']) for x in test}
    seen=set();pool=[]
    for x in ds['train']:
        key=normalize(x['sentence'])
        if key not in seen and key not in heldout:
            seen.add(key);pool.append(dict(id=int(x['idx']),text=x['sentence'],label=int(x['label'])))
    rng=np.random.default_rng(SEED)
    train=[]
    for label in [0,1]:
        group=[x for x in pool if x['label']==label]
        train += [group[int(i)] for i in rng.choice(len(group),2048,replace=False)]
    rng.shuffle(train)
    for name,rows in [('train',train),('test',test)]:
        path=out/(name+'.jsonl');path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        meta[name+'_count']=len(rows);meta[name+'_ids']=[x['id'] for x in rows]
        meta[name+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    meta['manual_ids']=[x['id'] for x in test[:8]]
    meta['source_train_count']=len(ds['train'])
    dump('data_manifest.json',meta)
    print('Prepared',len(train),len(test),flush=True)

def read(name):
    return [json.loads(x) for x in Path('data',name+'.jsonl').read_text().splitlines()]

def prefix(tok,text):
    return tok.apply_chat_template([{'role':'system','content':PROMPT},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)

class Collator:
    def __init__(self,tok):self.tok=tok
    def __call__(self,rows):
        length=max(len(x['input_ids']) for x in rows)
        return {k:torch.tensor([x[k]+[pad]*(length-len(x[k])) for x in rows])
                for k,pad in [('input_ids',self.tok.pad_token_id),('attention_mask',0),('labels',-100)]}

@torch.inference_mode()
def evaluate(model,tok,rows):
    model.eval();tok.padding_side='left';records=[]
    for start in range(0,len(rows),32):
        chunk=rows[start:start+32]
        inputs=tok([prefix(tok,x['text']) for x in chunk],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
        outputs=model.generate(**inputs,max_new_tokens=4,do_sample=False,use_cache=True,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        texts=tok.batch_decode(outputs[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for x,answer in zip(chunk,texts):
            norm=answer.strip().lower()
            pred={'negative':0,'positive':1}.get(norm,-1)
            records.append({**x,'answer':answer,'prediction':pred})
    truth=[r['label'] for r in records];pred=[r['prediction'] for r in records]
    return dict(accuracy=accuracy_score(truth,pred),macro_f1=f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0),
                invalid_answers=sum(x==-1 for x in pred),test_count=len(rows)),records

def run(name):
    set_seed(SEED);torch.set_num_threads(4)
    assert torch.cuda.device_count()==1,'Select one GPU with CUDA_VISIBLE_DEVICES'
    out=Path('results')/name;out.mkdir(parents=True,exist_ok=False)
    meta=json.loads(Path('data_manifest.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=meta['model_revision'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=meta['model_revision'],dtype=torch.bfloat16,attn_implementation='sdpa')
    if CONFIGS[name] is not None:
        model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',bias='none',**CONFIGS[name]))
    model.to('cuda')
    total=sum(x.numel() for x in model.parameters())
    trainable=sum(x.numel() for x in model.parameters() if x.requires_grad) if name!='base' else 0
    params=dict(num_train_epochs=1,per_device_train_batch_size=8,gradient_accumulation_steps=2,
        learning_rate=2e-5,lr_scheduler_type='linear',warmup_ratio=0.05,weight_decay=0.01,
        optim='adamw_torch_fused',bf16=True,tf32=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=20,save_strategy='no',
        eval_strategy='no',report_to='none',seed=SEED,data_seed=SEED,disable_tqdm=True,dataloader_num_workers=2)
    seconds=peak=reserved=0;steps=0
    if name!='base':
        tok.padding_side='right';encoded=[]
        for x in read('train'):
            prompt=prefix(tok,x['text']);pids=tok.encode(prompt,add_special_tokens=False)
            ids=tok.encode(prompt+['negative','positive'][x['label']]+tok.eos_token,add_special_tokens=False)
            assert ids[:len(pids)]==pids
            assert len(ids)<=256,'Increase the shared limit rather than dropping examples'
            encoded.append(dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]*len(pids)+ids[len(pids):]))
        model.config.use_cache=False
        trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(out/'checkpoints'),**params),
            train_dataset=Dataset.from_list(encoded),data_collator=Collator(tok))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        trainer.train();torch.cuda.synchronize();seconds=time.perf_counter()-started
        peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
        steps=trainer.state.global_step;dump(out/'history.json',trainer.state.log_history)
        del trainer;gc.collect();torch.cuda.empty_cache()
    metrics,records=evaluate(model,tok,read('test'))
    result=dict(name=name,model=MODEL,model_revision=meta['model_revision'],lora=CONFIGS[name],training=params,
        training_seconds=seconds,peak_allocated_gib=peak,peak_reserved_gib=reserved,optimizer_steps=steps,
        total_parameters=total,trainable_parameters=trainable,gpu=torch.cuda.get_device_name(),seed=SEED,**metrics)
    dump(out/'summary.json',result);dump(out/'predictions.json',records)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare',*CONFIGS]);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run(args.mode)
