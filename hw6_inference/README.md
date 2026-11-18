# ДЗ 6. Ускорение инференса

[Отчёт](report.md), [таблицы и графики](summary).

Скопируйте requests.jsonl из архива задания в папку data. Скрипт сверит SHA256 файла перед запуском. В запросах уже есть параметры генерации. run_series.sh запускает Qwen/Qwen3-8B на ревизии b968826d9c46dd6066d109eabc6255188de91218.

    docker build -t hw6-inference .
    export RESULTS_DIR=reruns
    bash run_series.sh 0 base tuning 120
    bash run_series.sh 0 prefill2048 tuning 120
    bash run_series.sh 0 prefill512 tuning 120
    bash run_series.sh 0 prefill16384 tuning 120
    bash run_series.sh 0 interactive tuning 120
    bash run_series.sh 0 base_ngram tuning 120

Первый аргумент run_series.sh задаёт номер GPU. Настройки vLLM лежат в serve.sh. benchmark.sh по умолчанию отправляет все 640 запросов, load_test.sh держит нагрузку 300 секунд. Оба скрипта сбрасывают кэш до и после замера.

Для сравнения со спекдеком берём по 160 одинаковых запросов на каждой нагрузке.

    REQUEST_COUNT=160 RESULT_CONFIG=plain bash run_series.sh 0 base spec 60 120 240 480 720 960
    REQUEST_COUNT=160 RESULT_CONFIG=spec bash run_series.sh 0 base_ngram spec 60 120 240 480 720 960
    REQUEST_COUNT=160 RESULT_CONFIG=plain_v1 bash run_series.sh 0 base_v1 spec_check 60 120 240

Запустите нагрузочный тест с interactivity.

    bash run_series.sh 0 interactive load 150 160 165 170 175 180
    RESULT_CONFIG=interactive_repeat bash run_series.sh 0 interactive load 170

Сборка таблиц и графиков из готовых замеров.

    python3 -m pip install matplotlib
    python3 summarize.py results --output summary

Для своих замеров укажите папку reruns.

    python3 summarize.py reruns --output rerun_summary
