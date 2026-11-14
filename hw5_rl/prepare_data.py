"""Turn the original ToolACE conversations into RL prompts and references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SEED = 42
MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def schema_types(value):
    if isinstance(value, list):
        return [schema_types(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: schema_types(item) for key, item in value.items() if item is not None}
    if isinstance(result.get("type"), str):
        result["type"] = {
            "dict": "object", "float": "number", "tuple": "array", "any": "string"
        }.get(result["type"], result["type"])
    return result


def tool_schemas(system: str):
    start = system.find("[{", system.find("Here is a list of functions"))
    if start < 0:
        raise ValueError("tool list absent")
    raw, _ = json.JSONDecoder().raw_decode(system[start:])
    tools = []
    names = {}
    occupied = set()
    for tool in raw:
        original = str(tool["name"])
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", original).strip("_") or "tool"
        if name[0].isdigit():
            name = "tool_" + name
        root = name
        n = 2
        while name in occupied:
            name = f"{root}_{n}"
            n += 1
        occupied.add(name)
        names[original] = name
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": schema_types(tool.get("parameters") or {"type": "object"}),
            },
        })
    return tools, names


def literal(node):
    if isinstance(node, ast.Name):
        values = {"true": True, "false": False, "null": None}
        if node.id in values:
            return values[node.id]
    if isinstance(node, ast.List):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def split_calls(body: str):
    parts = []
    depth = 0
    quote = None
    escaped = False
    start = 0
    for index, char in enumerate(body):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced calls")
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    if quote or depth != 0:
        raise ValueError("unbalanced calls")
    parts.append(body[start:].strip())
    return [x for x in parts if x]


def parse_reference(text: str, names: dict[str, str]):
    text = text.strip()
    if not text.startswith("["):
        return []
    if not text.endswith("]"):
        raise ValueError("incomplete call list")
    result = []
    for part in split_calls(text[1:-1]):
        left = part.find("(")
        if left < 1 or not part.endswith(")"):
            raise ValueError("invalid call")
        name = part[:left].strip()
        if name not in names:
            raise ValueError("unlisted function")
        parsed = ast.parse("f" + part[left:], mode="eval").body
        if not isinstance(parsed, ast.Call) or parsed.args or any(k.arg is None for k in parsed.keywords):
            raise ValueError("unsupported arguments")
        result.append({
            "name": names[name],
            "arguments": {k.arg: literal(k.value) for k in parsed.keywords},
        })
    return result


def normalized_question(value: str):
    return " ".join(value.casefold().split())


def prepare(raw: list, benchmark: list, tokenizer=None, max_prompt_tokens=1536):
    bfcl_questions = {
        normalized_question(message.get("content", ""))
        for example in benchmark
        for message in example["question"]
        if message.get("role") == "user"
    }
    counters = Counter()
    rows = []
    lengths = []
    examples = {"call": None, "no_call": None}
    for dialogue_id, dialogue in enumerate(raw):
        counters["dialogues"] += 1
        try:
            tools, names = tool_schemas(dialogue["system"])
            turns = dialogue["conversations"]
            user = next(x["value"] for x in turns if x["from"] == "user")
            answer = next(x["value"] for x in turns if x["from"] == "assistant")
            if normalized_question(user) in bfcl_questions:
                counters["bfcl_overlap_removed"] += 1
                continue
            calls = parse_reference(answer, names)
            if not calls and not answer.strip():
                raise ValueError("empty no-call answer")
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if tokens > max_prompt_tokens:
                    counters["too_long"] += 1
                    continue
            else:
                prompt = None
                tokens = None
            label = "call" if calls else "no_call"
            counters[label] += 1
            lengths.append({"question_chars": len(user), "answer_chars": len(answer), "tools": len(tools), "prompt_tokens": tokens})
            if examples[label] is None:
                examples[label] = {"question": user[:250], "answer": answer[:250], "tools": len(tools)}
            rows.append({
                "source_id": dialogue_id,
                "prompt": prompt,
                "expected_calls": json.dumps(calls, ensure_ascii=False),
                "label": label,
                "tool_count": len(tools),
            })
        except (KeyError, ValueError, SyntaxError, TypeError, StopIteration) as exc:
            counters[f"invalid_{type(exc).__name__}"] += 1
    return rows, counters, lengths, examples


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {str(q): values[int((len(values) - 1) * q)] for q in (0, .5, .9, .99, 1)}


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("toolace_data.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("bfcl.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-limit", type=int, default=300)
    args = parser.parse_args()
    raw_bytes = args.source.read_bytes()
    raw = json.loads(raw_bytes)
    benchmark = [json.loads(line) for line in args.benchmark.read_text().splitlines()]
    tokenizer = None
    if not args.preview:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    rows, counters, lengths, examples = prepare(raw, benchmark, tokenizer, args.max_prompt_tokens)
    random.Random(SEED).shuffle(rows)
    val_ids = {row["source_id"] for row in rows[:round(len(rows) * .1)]}
    train = [row for row in rows if row["source_id"] not in val_ids][:args.train_limit]
    validation = [row for row in rows if row["source_id"] in val_ids][:args.validation_limit]
    call_train = [row for row in train if row["label"] == "call"]
    no_call_train = [row for row in train if row["label"] == "no_call"]
    if len(call_train) < 2000 or not no_call_train:
        raise ValueError("not enough source examples for the balanced RL experiment")
    balance_rng = random.Random(SEED + 1)
    selected_no_call = balance_rng.choices(no_call_train, k=1000)
    balanced = call_train[:2000] + selected_no_call
    balance_rng.shuffle(balanced)
    summary = {
        "source": "Team-ACE/ToolACE",
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "all_dialogues": len(raw),
        "counts": dict(counters),
        "selected_train": len(train),
        "selected_validation": len(validation),
        "standard_train": {"call": len(call_train), "no_call": len(no_call_train)},
        "balanced_train": {"total": len(balanced), "call": 2000, "no_call": 1000, "distinct_no_call": len({row["source_id"] for row in selected_no_call})},
        "max_prompt_tokens": args.max_prompt_tokens,
        "quantiles": {key: quantiles([x[key] for x in lengths if x[key] is not None]) for key in lengths[0]} if lengths else {},
        "examples": examples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eda.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not args.preview:
        write_jsonl(args.output / "train.jsonl", train)
        write_jsonl(args.output / "train_balanced.jsonl", balanced)
        write_jsonl(args.output / "validation.jsonl", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
