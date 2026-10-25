# ДЗ 3. SFT для function calling

[Выполненный ноутбук](sft_function_calling_hw.ipynb) · [Метрики](outputs/eval_results.csv) · [График loss](outputs/loss.png)

Для повторного запуска создайте venv, установите requirements.txt и выберите ядро llm-hw3.

    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m ipykernel install --user --name llm-hw3
    CUDA_VISIBLE_DEVICES=0 .venv/bin/jupyter lab

Веса модели не включены в репозиторий. При выполнении ноутбука они сохранятся в outputs/full_sft.
