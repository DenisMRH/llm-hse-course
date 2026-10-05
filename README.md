# LLM: домашнее задание 1

[Исследовательский отчёт](hw1_report.md) · [Код](your_solution.py) · [Графики](results)

## Воспроизведение

В папке репозитория:

```bash
docker build -t llm-hw1 .
docker run --rm -it --gpus device=2 --shm-size=8g --memory=96g --cpus=12 -v "$PWD:/app" llm-hw1 bash
```

`device=2` — GPU, использованная на devbox25; на другой машине укажите свободную GPU.

Внутри контейнера:

```bash
python -u your_solution.py prepare
python -u your_solution.py suite
python your_solution.py plot
```

Подготовленные данные сохраняются в `output_dir/`, веса и полные логи — в `runs/`, графики — в `results/`. Датасет и веса не загружаются в GitHub.

Каждый из восьми экспериментов имеет лимит 900 секунд. Оценка, сохранение и генерация требуют дополнительного времени. `suite` пропускает завершённые запуски; перед повторным экспериментом используйте новое имя через `train --run NAME` либо отдельную рабочую папку.

Пример запуска с лучшей конфигурацией этой серии:

```bash
python -u your_solution.py train --run my_best --learning-rate 0.0003 --schedule cosine --batch-size 16 --accumulation 2 --optim adamw_torch_fused --compile
```

Параметры и результаты восьми запусков, а также три примера генераций приведены в `hw1_report.md`. В репозитории оставлены три графика, используемые в отчёте. Полные логи и служебные результаты сохраняются на машине запуска.
