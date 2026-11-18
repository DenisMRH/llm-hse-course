"""Build homework tables and plots from AIPerf v0.13.0 exports."""

import argparse
import csv
import json
import re
from pathlib import Path


def counter(path, name):
    pattern = re.compile(rf"^vllm:{name}_total(?:\{{[^}}]*\}})? (\S+)", re.M)
    return sum(float(value) for value in pattern.findall(path.read_text())) if path.is_file() else 0


def read_run(path, root):
    phase, config, rpm, _, _ = path.relative_to(root).parts
    profile = json.loads(path.read_text())
    latency = profile.get("request_latency", {})
    scale = {"ms": 0.001, "s": 1}.get(latency.get("unit"))
    if latency and scale is None:
        raise ValueError(f"{path}: unexpected latency unit {latency.get('unit')}")
    mean = latency.get("avg")
    p90 = latency.get("p90")
    complete = profile.get("is_complete") is not False and not profile.get("was_cancelled")
    errors = profile.get("error_request_count", {}).get("avg", 0)
    successes = profile.get("request_count", {}).get("avg", latency.get("count", 0))
    phase_configs = (profile.get("input_config") or {}).get("phases", [])
    duration = next(
        (item.get("duration") for item in phase_configs if item.get("name") == "profiling"),
        None,
    )
    before = path.parents[2] / f"metrics-{rpm}-before.txt"
    after = path.parents[2] / f"metrics-{rpm}-after.txt"
    draft_tokens = counter(after, "spec_decode_num_draft_tokens") - counter(before, "spec_decode_num_draft_tokens")
    accepted = counter(after, "spec_decode_num_accepted_tokens") - counter(before, "spec_decode_num_accepted_tokens")
    return {
        "phase": phase,
        "config": config,
        "rpm": int(rpm),
        "mean_s": mean * scale if mean is not None else None,
        "p90_s": p90 * scale if p90 is not None else None,
        "ttft_mean_ms": profile.get("time_to_first_token", {}).get("avg"),
        "itl_mean_ms": profile.get("inter_token_latency", {}).get("avg"),
        "output_tokens_mean": profile.get("output_sequence_length", {}).get("avg"),
        "accepted_draft_pct_with_warmup": 100 * accepted / draft_tokens if draft_tokens else None,
        "successes": successes,
        "errors": errors,
        "complete": complete,
        "effective_rpm": profile.get("request_throughput", {}).get("avg", 0) * 60,
        "configured_duration_s": duration,
        "sent_rpm": 60 * (successes + errors) / duration if duration else None,
        "export": str(path.relative_to(root)),
    }


def valid(row):
    return row["complete"] and not row["errors"] and row["p90_s"] is not None


def plot_spec(rows, output):
    import matplotlib.pyplot as plt

    plain = {row["rpm"]: row for row in rows if row["config"] == "plain" and valid(row)}
    spec = {row["rpm"]: row for row in rows if row["config"] == "spec" and valid(row)}
    rpms = sorted(plain.keys() & spec.keys())
    if not rpms:
        return
    fig, (latency, speedup) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    latency.plot(rpms, [plain[r]["p90_s"] for r in rpms], "o-", label="Без спекдека")
    latency.plot(rpms, [spec[r]["p90_s"] for r in rpms], "s-", label="Со спекдеком")
    latency.set_ylabel("P90 E2E, с")
    latency.legend()
    speedup.plot(rpms, [plain[r]["p90_s"] / spec[r]["p90_s"] for r in rpms], "o-")
    speedup.axhline(1, color="gray", linestyle="--")
    speedup.set(xlabel="Нагрузка, RPM", ylabel="Ускорение по P90", xticks=rpms)
    for axis in (latency, speedup):
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "spec_vs_rpm.png", dpi=180)
    plt.close(fig)
    for rpm in rpms:
        if spec[rpm]["p90_s"] > plain[rpm]["p90_s"]:
            print(f"Первый замер, где P90 со спекдеком выше: {rpm} RPM")
            break
    else:
        print("На измеренных нагрузках P90 со спекдеком не выше P90 без него.")


def plot_load(rows, output, best):
    import matplotlib.pyplot as plt

    rows = [
        row for row in rows
        if row["config"] in (best, f"{best}_repeat") and row["configured_duration_s"] == 300
        and row["complete"] and row["p90_s"] is not None
    ]
    if not rows:
        return
    primary = [row for row in rows if row["config"] == best]
    repeats = [row for row in rows if row["config"] != best]
    last_pass, first_fail = None, None
    for rpm in sorted({row["rpm"] for row in rows}):
        runs = [r for r in rows if r["rpm"] == rpm]
        slowest = max(runs, key=lambda r: r["p90_s"])
        if all(valid(r) and r["p90_s"] < 15 and r["sent_rpm"] >= rpm - 1 for r in runs):
            last_pass = slowest
        else:
            first_fail = slowest
            break
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot([r["rpm"] for r in primary], [r["mean_s"] for r in primary], "o-", label="Среднее E2E")
    axis.plot([r["rpm"] for r in primary], [r["p90_s"] for r in primary], "s-", label="P90 E2E")
    if repeats:
        axis.scatter([r["rpm"] for r in repeats], [r["p90_s"] for r in repeats],
                     marker="D", color="purple", label="Повтор P90", zorder=4)
    axis.axhline(15, color="gray", linestyle="--", label="SLA 15 с")
    for row, label, color in ((last_pass, "Предел в замерах", "green"), (first_fail, "Выход за SLA", "red")):
        if row:
            axis.scatter(row["rpm"], row["p90_s"], color=color, s=65, label=label, zorder=3)
            axis.annotate(f"{row['rpm']} RPM", (row["rpm"], row["p90_s"]), xytext=(0, 10),
                          textcoords="offset points", ha="center", color=color)
            print(f"{label}: {row['rpm']} RPM, P90 {row['p90_s']:.3f} с")
    axis.set(xlabel="Нагрузка, RPM", ylabel="Задержка, с", xticks=sorted({r["rpm"] for r in rows}))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "load_vs_latency.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results/<tuning|spec|spec_check|load>/<config>/<rpm>/")
    parser.add_argument("--output", type=Path, default=Path("summary"))
    args = parser.parse_args()
    rows = sorted(
        (read_run(path, args.results) for path in args.results.glob("*/*/*/profile/profile_export_aiperf.json")),
        key=lambda row: (row["phase"], row["rpm"], row["config"]),
    )
    if not rows:
        parser.error("no AIPerf exports found")
    tuning = [r for r in rows if r["phase"] == "tuning" and r["rpm"] == 120]
    comparable = [r for r in tuning if valid(r) and r["mean_s"] is not None]
    baseline = next((r for r in comparable if r["config"] == "base"), None)
    for row in rows:
        row["speedup_at_120"] = (
            baseline["mean_s"] / row["mean_s"] if baseline and row in tuning and row["mean_s"] else None
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for phase in ("tuning", "spec", "spec_check", "load"):
        points = [row for row in rows if row["phase"] == phase]
        if points:
            with (args.output / f"{phase}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=points[0].keys(), lineterminator="\n")
                writer.writeheader()
                writer.writerows(points)
    plot_spec([row for row in rows if row["phase"] == "spec"], args.output)
    if baseline:
        best = min(comparable, key=lambda row: row["mean_s"])
        print(f"Лучшая конфигурация при 120 RPM: {best['config']}, {best['speedup_at_120']:.3f} раза")
        plot_load([row for row in rows if row["phase"] == "load"], args.output, best["config"])


if __name__ == "__main__":
    main()
