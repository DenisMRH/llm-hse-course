"""Create a short PDF report strictly from saved experiment artifacts."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hw5_rl_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)


ROOT = Path(__file__).resolve().parent
NAMES = ["rl1_grpo", "rl2_early_stop", "rl3_balanced", "rl4_balanced_concise"]
LABELS = ["Исходная", "RL-1: GRPO", "RL-2: 40 шагов", "RL-3: баланс", "RL-4: краткий ответ"]
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
if not (FONT_DIR / "Arial.ttf").exists():
    FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def read_json(path):
    return json.loads(path.read_text())


def metric(history, kind):
    for key in (kind, f"train/{kind}", f"rewards/dense_reward/mean", "reward/mean"):
        points = [(row.get("step", 0), row[key]) for row in history if key in row and row.get("step") is not None]
        if points:
            return points, key
    raise ValueError(f"No {kind} values in training history")


def training_chart(runs):
    figure, axes = plt.subplots(4, 2, figsize=(11.5, 9.0), sharex="col")
    for row, name in enumerate(NAMES):
        history = read_json(ROOT / "runs" / name / "history.json")
        for column, kind in enumerate(("loss", "reward")):
            points, _ = metric(history, kind)
            ax = axes[row, column]
            ax.plot([p[0] for p in points], [p[1] for p in points], color="#185a9d", lw=1.4)
            ax.grid(alpha=.2)
            ax.set_title(f"{LABELS[row + 1]}: {kind}", fontsize=10)
            ax.set_ylabel("Loss" if kind == "loss" else "Reward")
            if row == 3:
                ax.set_xlabel("Training step")
    figure.tight_layout()
    output = ROOT / "figures" / "training_curves.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def error_examples(best):
    benchmark = {row["id"]: row for row in map(json.loads, (ROOT / "bfcl.jsonl").read_text().splitlines())}
    predictions = [json.loads(line) for line in (ROOT / "results" / best / "predictions.jsonl").read_text().splitlines()]
    wrong = [row for row in predictions if not row["check"]["valid"]]

    def choose(test, used, shortest=False):
        candidates = wrong
        if shortest:
            candidates = sorted(wrong, key=lambda row: len(" ".join(
                message.get("content", "") for message in benchmark[row["id"]]["question"]
                if message.get("role") == "user"
            )))
        for row in candidates:
            if row["id"] not in used and test(row):
                used.add(row["id"])
                return row
        return None

    used = set()
    picked = [
        ("Лишний вызов", choose(lambda x: "irrelevance" in x["category"] and bool(x["calls"]), used)),
        ("Пропущенный вызов", choose(lambda x: "irrelevance" not in x["category"] and not x["calls"], used)),
        ("Параллельные вызовы", choose(lambda x: "parallel" in x["category"], used, shortest=True)),
        ("Неверные аргументы", choose(lambda x: "irrelevance" not in x["category"] and bool(x["calls"]), used)),
    ]
    output = []
    for label, row in picked:
        if row is None:
            continue
        source = benchmark[row["id"]]
        question = " ".join(x.get("content", "") for x in source["question"] if x.get("role") == "user")
        expected = [next(iter(x)) for x in source["ground_truth"]]
        actual = [next(iter(x)) for x in row["calls"]]
        raw_error = row["check"].get("error", "")
        if isinstance(raw_error, list) and len(raw_error) == 1 and isinstance(raw_error[0], str):
            raw_error = raw_error[0]
        output.append({
            "label": label, "id": row["id"], "category": row["category"],
            "question": question[:190], "expected": expected, "actual": actual,
            "expected_count": len(source["ground_truth"]), "actual_count": len(row["calls"]),
            "error": str(raw_error)[:160].rstrip("."),
        })
    return output


def truncated_irrelevance(name):
    rows = [json.loads(line) for line in (ROOT / "results" / name / "predictions.jsonl").read_text().splitlines()]
    return sum("irrelevance" in row["category"] and row["finish_reason"] == "length" for row in rows)


def main():
    regular = FONT_DIR / ("Arial.ttf" if (FONT_DIR / "Arial.ttf").exists() else "DejaVuSans.ttf")
    bold = FONT_DIR / ("Arial Bold.ttf" if (FONT_DIR / "Arial Bold.ttf").exists() else "DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("ArialCustom", str(regular)))
    pdfmetrics.registerFont(TTFont("ArialCustomBold", str(bold)))
    pdfmetrics.registerFontFamily("ArialCustom", normal="ArialCustom", bold="ArialCustomBold")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleRu", parent=styles["Title"], fontName="ArialCustomBold", fontSize=16, leading=21, alignment=TA_CENTER, textColor=colors.HexColor("#17324d"), spaceAfter=15)
    heading = ParagraphStyle("HeadingRu", parent=styles["Heading2"], fontName="ArialCustomBold", fontSize=11, leading=15, textColor=colors.HexColor("#17324d"), spaceBefore=13, spaceAfter=6)
    body = ParagraphStyle("BodyRu", parent=styles["BodyText"], fontName="ArialCustom", fontSize=9.3, leading=13.5, spaceAfter=6)
    small = ParagraphStyle("SmallRu", parent=body, fontSize=8.4, leading=11.5)
    cell = ParagraphStyle("CellRu", parent=body, fontSize=8.1, leading=10.5, spaceAfter=0)

    def p(text, style=body):
        return Paragraph(html.escape(str(text)), style)

    eda = read_json(ROOT / "data" / "eda.json")
    metrics = [read_json(ROOT / "results" / name / "metrics.json") for name in ("base", *NAMES)]
    runs = [read_json(ROOT / "runs" / name / "training.json") for name in NAMES]
    chart = training_chart(runs)
    best_index = max(range(len(metrics)), key=lambda i: metrics[i]["overall"])
    best_name = ("base", *NAMES)[best_index]
    reward_gain = (metrics[4]["overall"] - metrics[3]["overall"]) * 100
    last_rewards = []
    for name in NAMES:
        points, _ = metric(read_json(ROOT / "runs" / name / "history.json"), "reward")
        last_rewards.append(sum(value for _, value in points[-20:]) / min(20, len(points)))
    same_reward_runs = range(3)
    reward_rank_leader = max(same_reward_runs, key=lambda i: last_rewards[i])
    bfcl_rank_leader = max(same_reward_runs, key=lambda i: metrics[i + 1]["overall"])
    rl1_history = read_json(ROOT / "runs" / NAMES[0] / "history.json")
    rl1_rewards = [row["reward"] for row in rl1_history if "reward" in row]

    story = [p("ДЗ 5. RL для вызова функций", title)]
    story.append(p("1. Данные", heading))
    counts = eda["counts"]
    story.append(p(f"Из {eda['all_dialogues']} диалогов ToolACE взял первый ответ ассистента после первого запроса. Получилось {counts['call']} примеров с вызовом функции и {counts['no_call']} без вызова. Отсеял {counts.get('invalid_ValueError', 0) + counts.get('invalid_SyntaxError', 0)} записей с ошибкой разбора и {counts.get('too_long', 0)} слишком длинных промптов."))
    q = eda["quantiles"]
    story.append(p(f"Медиана длины промпта {q['prompt_tokens']['0.5']} токенов, 90-й процентиль {q['prompt_tokens']['0.9']}; медиана числа доступных функций {q['tools']['0.5']}. Имена функций привёл к формату Qwen, описания и аргументы сохранил."))
    story.append(p(f"Разделил данные по исходным диалогам с сидом 42 и выделил 10% на валидацию. Для обучения взял {eda['selected_train']} примеров, для проверки {eda['selected_validation']}. Совпадений запросов с BFCL не нашёл. В обычной обучающей выборке {eda['standard_train']['call']} вызовов и {eda['standard_train']['no_call']} ответов без вызова. Для RL-3 и RL-4 взял {eda['balanced_train']['call']} вызовов и {eda['balanced_train']['no_call']} ответов без вызова, последние с повторениями из {eda['balanced_train']['distinct_no_call']} диалогов."))
    call_sample = eda["examples"]["call"]
    no_call_sample = eda["examples"]["no_call"]
    story.append(p(f"Пример с вызовом: {call_sample['question']} Эталон: {call_sample['answer']}", small))
    story.append(p(f"Пример без вызова: {no_call_sample['question']} Эталон: текстовый ответ без обращения к инструменту.", small))

    story.append(p("2. Награда и конфигурации", heading))
    story.append(p("За точный набор вызовов с верными аргументами даю 1. При частичном совпадении к 0,1 добавляю 0,4 F1 по именам функций и 0,5 F1 по аргументам. Порядок параллельных вызовов не важен. Если вызов не нужен, непустой ответ без него получает 1, вызов получает 0. За некорректную разметку даю 0."))
    story.append(p("В RL-4 поменял награду за ответ без вызова на exp(-max(0, L - 240) / 500), где L - число символов. За достижение лимита в 256 токенов дополнительно умножаю её на 0,2. Награда за вызовы прежняя."))
    story.append(p("Пример награды: get_weather(city=Paris, days=2) при точном совпадении получает 1; другой город даёт частичную награду; незакрытый тег вызова получает 0. Эталоны ToolACE нужны только для вычисления награды. Целевые ответы модель при обучении не видит."))
    story.append(p("Все запуски используют Qwen/Qwen3.5-2B и GRPO из TRL. LoRA: r 16, alpha 32, dropout 0, модули q, k, v, o, gate, up, down. Скорость обучения 5e-5, cosine scheduler, 5 шагов прогрева, 4 генерации на запрос, batch 8, accumulation 2, максимум 256 токенов, температура 0,8, beta 0. Перед оценкой веса LoRA объединены с моделью."))
    config_table = [[p(x, cell) for x in ("Запуск", "Конфигурация", "Гипотеза")]]
    changes = ["120 шагов, обычная выборка", "40 шагов, обычная выборка", "120 шагов, 33% без вызова", "120 шагов, новая награда"]
    hypotheses = ["Частичная награда улучшит вызовы.", "Меньше шагов сохранит ответы без вызова.", "Баланс уменьшит число лишних вызовов.", "Штраф сократит обрезанные ответы."]
    for i, name in enumerate(NAMES):
        config_table.append([p(LABELS[i + 1], cell), p(changes[i], cell), p(hypotheses[i], cell)])
    table = Table(config_table, colWidths=[95, 170, 225], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f7")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#ccd7e0")), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(table)
    story.append(Spacer(1, 9))
    story.append(Image(str(chart), width=510, height=399))
    story.append(p("Loss и средняя награда на каждом шаге обучения.", small))

    story.append(p("3. Результаты BFCL", heading))
    header = ["Модель", "Overall", "Simple", "Multiple", "Parallel", "Parallel multiple", "Irrel."]
    rows = [[p(x, cell) for x in header]]
    for label, result in zip(LABELS, metrics):
        group = result["groups"]
        rows.append([p(label, cell), *[p(f"{x:.1%}", cell) for x in (result["overall"], group["simple"], group["multiple"], group["parallel"], group["parallel_multiple"], group["irrelevance"])]])
    table = Table(rows, colWidths=[114, 58, 58, 63, 63, 93, 59], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f7")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#ccd7e0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)
    story.append(p("Оценка по одному BFCL из 350 примеров, итоговый overall из evaluate.py. Выбранные чекпоинты: " + ", ".join(f"RL-{i+1}, шаг {run['step']}" for i, run in enumerate(runs)) + ".", small))

    story.append(PageBreak())
    story.append(p("4. Ответы на вопросы задания", heading))
    story.append(p(f"Частичная награда помогает учиться на близких к эталону вызовах. Лучше всех сработал {LABELS[best_index]}: {metrics[best_index]['overall']:.1%} overall, на {(metrics[best_index]['overall']-metrics[0]['overall'])*100:.1f} п.п. выше исходной модели. Самое заметное изменение между RL-3 и RL-4: штраф за длинный ответ без вызова дал {reward_gain:.1f} п.п. Остальные настройки в этой паре одинаковы. В RL-2 число шагов меняет и расписание скорости обучения."))
    story.append(p(f"Рост обучающей награды не совпал с BFCL. У RL-1 средняя награда выросла с {sum(rl1_rewards[:20])/20:.3f} за первые 20 шагов до {sum(rl1_rewards[-20:])/20:.3f} за последние 20, а BFCL упал ниже исходного. Среди RL-1, RL-2 и RL-3 по награде лидирует {LABELS[reward_rank_leader+1]}, по BFCL - {LABELS[bfcl_rank_leader+1]}. Награду RL-4 с ними напрямую не сравниваю: её формула другая."))
    truncated = [truncated_irrelevance(name) for name in ("base", *NAMES)]
    story.append(p("Обрезанные ответы в категории Irrelevance из 50: " + ", ".join(f"{LABELS[i]} - {count}" for i, count in enumerate(truncated)) + ". В RL-1 обрезаний стало больше, хотя обучающая награда выросла: исходная награда не учитывала длину ответа. После штрафа в RL-4 обрезаний нет."))
    story.append(p("При одинаковой награде всех четырёх ответов на запрос их advantage равен нулю. Этот запрос не влияет на обновление модели."))
    story.append(p("Примеры ошибок лучшей модели", heading))
    def calls_word(count):
        if count % 10 == 1 and count % 100 != 11:
            return "вызов"
        if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
            return "вызова"
        return "вызовов"

    for example in error_examples(best_name):
        expected = ", ".join(example["expected"]) or "нет вызова"
        actual = ", ".join(example["actual"]) or "нет вызова"
        text = (f"{example['label']} ({example['id']}): {example['question']} "
                f"Ожидалось: {example['expected_count']} {calls_word(example['expected_count'])} ({expected}). "
                f"Получено: {example['actual_count']} {calls_word(example['actual_count'])} ({actual}). "
                f"Оценщик: {example['error']}.")
        story.append(p(text, small))

    output = ROOT / "report.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=43, leftMargin=43, topMargin=42, bottomMargin=43, title="ДЗ 5. RL для вызова функций")
    def footer(canvas, document):
        canvas.setFont("ArialCustom", 8)
        canvas.drawRightString(A4[0] - 43, 25, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
