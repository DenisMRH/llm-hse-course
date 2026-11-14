# ДЗ 5. RL для вызова функций

[Отчёт](report.pdf), [подготовка данных](prepare_data.py), [награда](reward.py), [обучение и экспорт](train.py), [конфигурации](configs), [результаты](results).

Воспроизведение на машине с NVIDIA GPU и Docker:

    GPU_INDEX=0 bash run_all.sh

Построение отчёта:

    python3 -m pip install reportlab matplotlib
    python3 make_report.py

Сид: 42. Модель: Qwen/Qwen3.5-2B, ревизия 15852e8c16360a2fea060d615a32b45270f8a8fc. Данные: Team-ACE/ToolACE, ревизия 6bda777c88d21e5a204703c1ee45597a8fa4f734. Версии библиотек закреплены в [Dockerfile](Dockerfile). Выданные [BFCL](bfcl.jsonl) и [оценщик](evaluate.py) включены в папку.

Логи обучения и выбранный шаг: [runs](runs). Метрики и ответы исходной модели и каждого запуска: [results](results).
