"""
Summarise the instance-split vs question-split re-evaluation of the
within-condition conflict detector and CAS.

Reads run directories written by run_conflict_prediction.py under
results_repro/:

  <mode>_seed<k>/                     full feature set
  ablation_<feature_set>_<mode>_seed<k>/  feature-set ablations (IC, ICC)

and writes results_repro/split_comparison.md, results_repro/split_comparison.csv
and figures_repro/cas_lift_instance_vs_question.png.

Usage
-----
  python summarize_split_comparison.py
  python summarize_split_comparison.py --repro-dir results_repro --fig-dir figures_repro
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = ["llama3.1-8b", "meditron3-8b", "phi4-14b", "qwen3-4b", "qwen3-8b", "qwen3.5-9b"]
CONDITIONS = ["NC", "CC", "IC", "CIC", "ICC"]
PRIORITY = ["IC", "ICC"]
MODES = ["instance", "question"]
COVERAGES = ["0.75", "0.5", "0.25"]
FEATURE_SETS = ["full", "unc", "unc_docs", "question"]
MEAN = "6-model mean"

RUN_RE = re.compile(r"^(?:ablation_(?P<fs>unc_docs|unc|question)_)?(?P<mode>instance|question)_seed(?P<seed>\d+)$")


# ── Loading ──────────────────────────────────────────────────────────
def load_runs(repro_dir: Path, seeds: set[int] | None = None) -> list[dict]:
    """Return one record per (run dir, model, condition), restricted to ``seeds``."""
    records = []
    for run_dir in sorted(repro_dir.iterdir()):
        m = RUN_RE.match(run_dir.name) if run_dir.is_dir() else None
        if not m:
            continue
        fs = m.group("fs") or "full"
        mode, seed = m.group("mode"), int(m.group("seed"))
        if seeds is not None and seed not in seeds:
            continue
        for model in MODELS:
            path = run_dir / f"{model}_conflict_prediction.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for cond, res in data["conditions"].items():
                wc = res["within_condition"]
                dm = wc["detector_metrics"]
                rec = {
                    "run": run_dir.name, "feature_set": fs, "mode": mode,
                    "seed": seed, "model": model, "condition": cond,
                    "auroc_detector": _num(dm["auroc_conflict_score"]),
                    "auroc_confidence": _num(dm["auroc_confidence"]),
                    "auroc_combined": _num(wc["signal_comparison"]["combined"]["auroc"]),
                    "split_info": wc.get("split_info") or {},
                    "n_cv_folds": len(dm["fold_aurocs"]),
                    "n_test": res["n_test"],
                }
                for cov in COVERAGES:
                    sp = wc["selective_prediction"].get(cov)
                    if sp is None:
                        continue
                    rec[f"acc_cas_{cov}"] = sp["test_accuracy_combined"]
                    rec[f"acc_conf_{cov}"] = sp["test_accuracy_conf_only"]
                    rec[f"cov_cas_{cov}"] = sp["test_coverage_combined"]
                    rec[f"cov_conf_{cov}"] = sp["test_coverage_conf_only"]
                    rec[f"lift_{cov}"] = sp["test_accuracy_combined"] - sp["test_accuracy_conf_only"]
                    ex = wc.get("selective_prediction_exact_topk", {}).get(cov)
                    if ex is not None:
                        rec[f"xacc_cas_{cov}"] = ex["test_accuracy_combined"]
                        rec[f"xacc_conf_{cov}"] = ex["test_accuracy_conf_only"]
                        rec[f"xlift_{cov}"] = ex["test_accuracy_combined"] - ex["test_accuracy_conf_only"]
                records.append(rec)
    return records


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def is_degenerate(rec: dict) -> bool:
    """Split where the test or train labels were single-class, or CV was skipped."""
    si = rec["split_info"]
    return (
        math.isnan(rec["auroc_detector"])
        or rec["n_cv_folds"] == 0
        or si.get("n_test_positive_labels", 1) == 0
        or si.get("n_train_positive_labels", 1) == 0
        or si.get("split_method") == "GroupShuffleSplit"
    )


# ── Aggregation ──────────────────────────────────────────────────────
def aggregate(records, metric, feature_set, mode, condition):
    """Per model and six-model mean: (mean, sd, n_seeds) over seeds.

    The six-model mean is computed per seed first, then summarised over seeds.
    """
    by_model_seed = defaultdict(dict)
    for r in records:
        if (r["feature_set"], r["mode"], r["condition"]) == (feature_set, mode, condition) and metric in r:
            by_model_seed[r["model"]][r["seed"]] = r[metric]
    out = {}
    for model in MODELS:
        vals = np.array(list(by_model_seed.get(model, {}).values()), dtype=float)
        out[model] = _stats(vals)
    seeds = set.intersection(*[set(by_model_seed.get(m, {})) for m in MODELS]) if by_model_seed else set()
    seed_means = np.array([np.nanmean([by_model_seed[m][s] for m in MODELS]) for s in sorted(seeds)])
    out[MEAN] = _stats(seed_means)
    return out


def _stats(vals: np.ndarray):
    vals = vals[~np.isnan(vals)] if len(vals) else vals
    if len(vals) == 0:
        return (float("nan"), float("nan"), 0)
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return (float(np.mean(vals)), sd, len(vals))


def fmt(stat, pct=False, signed=False):
    mean, sd, n = stat
    if n == 0 or math.isnan(mean):
        return "n/a"
    scale = 100 if pct else 1
    spec = ("+" if signed else "") + (".1f" if pct else ".3f")
    s = format(mean * scale, spec)
    if n > 1 and not math.isnan(sd):
        s += " ± " + format(sd * scale, ".1f" if pct else ".3f")
    return s


# ── Report ───────────────────────────────────────────────────────────
def build_report(records, repro_dir: Path, csv_rows: list):
    lines = []
    w = lines.append
    full = [r for r in records if r["feature_set"] == "full"]
    seeds = {m: sorted({r["seed"] for r in full if r["mode"] == m}) for m in MODES}
    conds_present = [c for c in CONDITIONS if any(r["condition"] == c for r in full)]

    w("# Instance split vs question split: conflict detector and CAS")
    w("")
    w("Within-condition detector, alpha = 0.5, tau = 0.7, thresholds chosen on train OOF scores and applied to test.")
    w("Values are mean ± SD over seeds. The six-model mean is taken per seed, then summarised over seeds.")
    w("")
    for m in MODES:
        w(f"- {m} split: {len(seeds[m])} seeds ({_seed_range(seeds[m])}).")
    w("")

    # Split diagnostics
    w("## Split diagnostics")
    w("")
    w("| Mode | Condition | Test fraction | Test questions | Questions shared train/test | Degenerate splits |")
    w("|---|---|---|---|---|---|")
    for mode in MODES:
        for cond in conds_present:
            rs = [r for r in full if r["mode"] == mode and r["condition"] == cond]
            if not rs:
                continue
            tf = np.array([r["split_info"].get("test_fraction", r["n_test"] / 920) for r in rs])
            tq = np.array([r["split_info"].get("n_test_questions", np.nan) for r in rs], dtype=float)
            sq = np.array([r["split_info"].get("n_shared_questions", np.nan) for r in rs], dtype=float)
            deg = sum(is_degenerate(r) for r in rs)
            w(f"| {mode} | {cond} | {np.mean(tf):.3f} [{np.min(tf):.3f}, {np.max(tf):.3f}] "
              f"| {np.nanmean(tq):.1f} | {np.nanmean(sq):.1f} | {deg} of {len(rs)} model-seed runs |")
    w("")

    # 1. AUROC
    w("## 1. Error-detection AUROC")
    w("")
    w("AUROC of the detector score r and of raw confidence for detecting wrong answers on the test split.")
    w("")
    for cond in conds_present:
        w(f"### {cond}")
        w("")
        w("| Model | Detector, instance | Detector, question | Confidence, instance | Confidence, question |")
        w("|---|---|---|---|---|")
        aggs = {(sig, mode): aggregate(full, f"auroc_{sig}", "full", mode, cond)
                for sig in ("detector", "confidence") for mode in MODES}
        for model in MODELS + [MEAN]:
            label = f"**{model}**" if model == MEAN else model
            w(f"| {label} | " + " | ".join(fmt(aggs[(sig, mode)][model])
                                       for sig in ("detector", "confidence") for mode in MODES) + " |")
            for (sig, mode), agg in aggs.items():
                csv_rows.append(_csv("auroc", cond, model, mode, "full", sig, "", agg[model]))
        w("")

    # 2-3. Selective prediction
    w("## 2. Selective accuracy at target coverage (CAS vs confidence)")
    w("")
    w("Accuracy in percent on the test split. Confidence is evaluated at the same number of answered "
      "instances as CAS. Lift = CAS accuracy minus confidence accuracy, in percentage points.")
    w("")
    for cond in conds_present:
        w(f"### {cond}")
        w("")
        w("| Model | Coverage | CAS, instance | Conf, instance | Lift, instance | CAS, question | Conf, question | Lift, question |")
        w("|---|---|---|---|---|---|---|---|")
        for model in MODELS + [MEAN]:
            label = f"**{model}**" if model == MEAN else model
            for cov in COVERAGES:
                cells = []
                for mode in MODES:
                    for metric in (f"acc_cas_{cov}", f"acc_conf_{cov}", f"lift_{cov}"):
                        st = aggregate(full, metric, "full", mode, cond)[model]
                        cells.append(fmt(st, pct=True, signed=metric.startswith("lift")))
                        csv_rows.append(_csv("selective", cond, model, mode, "full",
                                             metric.rsplit("_", 1)[0], cov, st))
                w(f"| {label} | {float(cov):.0%} | " + " | ".join(cells) + " |")
        w("")

    w("## 3. Realised test coverage")
    w("")
    w("Thresholds are chosen on train and transferred, so realised test coverage can differ from the target. "
      "Confidence coverage equals CAS coverage by construction except for ties.")
    w("")
    w("| Condition | Coverage target | CAS, instance | Conf, instance | CAS, question | Conf, question |")
    w("|---|---|---|---|---|---|")
    for cond in conds_present:
        for cov in COVERAGES:
            cells = []
            for mode in MODES:
                for sig in ("cas", "conf"):
                    st = aggregate(full, f"cov_{sig}_{cov}", "full", mode, cond)[MEAN]
                    cells.append(fmt(st, pct=True))
                    csv_rows.append(_csv("coverage", cond, MEAN, mode, "full", f"cov_{sig}", cov, st))
            w(f"| {cond} | {float(cov):.0%} | " + " | ".join(cells) + " |")
    w("")
    w("Per-model realised coverage is in the CSV.")
    for cond in conds_present:
        for cov in COVERAGES:
            for mode in MODES:
                for sig in ("cas", "conf"):
                    agg = aggregate(full, f"cov_{sig}_{cov}", "full", mode, cond)
                    for model in MODELS:
                        csv_rows.append(_csv("coverage", cond, model, mode, "full", f"cov_{sig}", cov, agg[model]))
    w("")

    # 3b. Exact top-k (matched coverage)
    w("## 3b. Matched-coverage check: exact top-k on test")
    w("")
    w("In the threshold-transfer protocol above, confidence-only keeps every instance tied at the confidence "
      "cut-off, so it answers more test instances than CAS (see section 3). As a check, this table uses the "
      "`selective_prediction_exact_topk` block that the same runs write: both signals answer exactly the top-k "
      "test instances. Six-model mean, percentage points.")
    w("")
    w("| Condition | Coverage | CAS, instance | Conf, instance | Lift, instance | CAS, question | Conf, question | Lift, question |")
    w("|---|---|---|---|---|---|---|---|")
    for cond in conds_present:
        for cov in COVERAGES:
            cells = []
            for mode in MODES:
                for metric in (f"xacc_cas_{cov}", f"xacc_conf_{cov}", f"xlift_{cov}"):
                    agg = aggregate(full, metric, "full", mode, cond)
                    cells.append(fmt(agg[MEAN], pct=True, signed=metric.startswith("xlift")))
                    for model in MODELS + [MEAN]:
                        csv_rows.append(_csv("selective_exact_topk", cond, model, mode, "full",
                                             metric.rsplit("_", 1)[0], cov, agg[model]))
            w(f"| {cond} | {float(cov):.0%} | " + " | ".join(cells) + " |")
    w("")

    # 4. Ablations
    abl = records
    abl_fs = [fs for fs in FEATURE_SETS if any(r["feature_set"] == fs for r in abl)]
    w("## 4. Feature-set ablations (detector AUROC)")
    w("")
    w("full = uncertainty + document embedding(s) + question embedding; unc = 3 uncertainty signals; "
      "unc_docs = uncertainty + document embedding(s); question = question embedding only.")
    w("")
    for cond in PRIORITY:
        w(f"### {cond}")
        w("")
        header = " | ".join(f"{fs}, {mode}" for fs in abl_fs for mode in MODES)
        w(f"| Model | {header} |")
        w("|---|" + "---|" * (len(abl_fs) * len(MODES)))
        aggs = {(fs, mode): aggregate(abl, "auroc_detector", fs, mode, cond) for fs in abl_fs for mode in MODES}
        for model in MODELS + [MEAN]:
            label = f"**{model}**" if model == MEAN else model
            w(f"| {label} | " + " | ".join(fmt(aggs[k][model]) for k in aggs) + " |")
            for (fs, mode), agg in aggs.items():
                if fs != "full":
                    csv_rows.append(_csv("ablation_auroc", cond, model, mode, fs, "detector", "", agg[model]))
        w("")
        w(f"CAS lift (pp) over confidence for {cond}, six-model mean:")
        w("")
        w("| Feature set | Mode | 75% | 50% | 25% |")
        w("|---|---|---|---|---|")
        for fs in abl_fs:
            for mode in MODES:
                cells = []
                for cov in COVERAGES:
                    st = aggregate(abl, f"lift_{cov}", fs, mode, cond)[MEAN]
                    cells.append(fmt(st, pct=True, signed=True))
                    if fs != "full":
                        csv_rows.append(_csv("ablation_selective", cond, MEAN, mode, fs, "lift", cov, st))
                w(f"| {fs} | {mode} | " + " | ".join(cells) + " |")
        w("")

    # Run A check against the cached paper results
    w("## Appendix: Run A (instance split, seed 42) vs cached paper results")
    w("")
    w(_run_a_check(repro_dir))
    return "\n".join(lines) + "\n"


def _seed_range(seeds):
    if not seeds:
        return "none"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ", ".join(map(str, seeds))


def _csv(section, cond, model, mode, fs, metric, cov, stat):
    mean, sd, n = stat
    return {"section": section, "condition": cond, "model": model, "split": mode,
            "feature_set": fs, "metric": metric, "coverage": cov,
            "mean": mean, "sd": sd, "n_seeds": n}


def _run_a_check(repro_dir: Path) -> str:
    cached_dir = repro_dir.parent / "results"
    run_dir = repro_dir / "instance_seed42"
    if not run_dir.exists():
        return "Run A directory not found.\n"
    out = ["| Condition | Model | Detector AUROC, paper | Detector AUROC, rerun | Max abs diff, selective acc/cov |",
           "|---|---|---|---|---|"]
    for cond in CONDITIONS:
        for model in MODELS:
            pc, rc = cached_dir / f"{model}_conflict_prediction.json", run_dir / f"{model}_conflict_prediction.json"
            if not (pc.exists() and rc.exists()):
                continue
            p = json.loads(pc.read_text(encoding="utf-8"))["conditions"].get(cond)
            r = json.loads(rc.read_text(encoding="utf-8"))["conditions"].get(cond)
            if p is None or r is None:
                continue
            pa = p["within_condition"]["detector_metrics"]["auroc_conflict_score"]
            ra = r["within_condition"]["detector_metrics"]["auroc_conflict_score"]
            diffs = []
            for cov in COVERAGES:
                ps, rs = p["within_condition"]["selective_prediction"][cov], r["within_condition"]["selective_prediction"][cov]
                for k in ("test_accuracy_combined", "test_coverage_combined",
                          "test_accuracy_conf_only", "test_coverage_conf_only"):
                    diffs.append(abs(ps[k] - rs[k]))
            out.append(f"| {cond} | {model} | {pa:.3f} | {ra:.3f} | {max(diffs):.3f} |")
    return "\n".join(out) + "\n"


# ── Figure ───────────────────────────────────────────────────────────
def plot_lift(records, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"instance": "#2a78d6", "question": "#eb6834"}
    labels = {"instance": "Instance split", "question": "Question split"}
    full = [r for r in records if r["feature_set"] == "full"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    x = np.arange(len(COVERAGES))
    width = 0.36
    for ax, cond in zip(axes, PRIORITY):
        for i, mode in enumerate(MODES):
            stats = [aggregate(full, f"lift_{cov}", "full", mode, cond)[MEAN] for cov in COVERAGES]
            means = np.array([s[0] for s in stats]) * 100
            sds = np.nan_to_num(np.array([s[1] for s in stats]) * 100)
            pos = x + (i - 0.5) * (width + 0.02)
            ax.bar(pos, means, width, color=colors[mode], label=labels[mode], zorder=2)
            ax.errorbar(pos, means, yerr=sds, fmt="none", ecolor="#52514e", elinewidth=1, capsize=3, zorder=3)
            for px, m, s in zip(pos, means, sds):
                ax.text(px, m + (s if m >= 0 else -s) + (0.6 if m >= 0 else -0.6), f"{m:+.1f}",
                        ha="center", va="bottom" if m >= 0 else "top", fontsize=8, color="#0b0b0b")
        ax.axhline(0, color="#52514e", linewidth=0.8, zorder=1)
        ax.margins(y=0.12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(c):.0%}" for c in COVERAGES])
        ax.set_xlabel("Target coverage")
        ax.set_title(cond, fontsize=11)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("CAS accuracy minus confidence\naccuracy (pp)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    n_inst = len({r["seed"] for r in full if r["mode"] == "instance"})
    n_q = len({r["seed"] for r in full if r["mode"] == "question"})
    fig.suptitle(f"CAS lift over confidence, six-model mean (error bars: SD over seeds; "
                 f"instance n={n_inst}, question n={n_q})", fontsize=10)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repro-dir", default="results_repro")
    ap.add_argument("--fig-dir", default="figures_repro")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 19], metavar=("FIRST", "LAST"),
                    help="Inclusive seed range to aggregate (default: 0 19). "
                         "instance_seed42 (Run A) is reported separately.")
    args = ap.parse_args()
    repro_dir, fig_dir = Path(args.repro_dir), Path(args.fig_dir)

    records = load_runs(repro_dir, set(range(args.seeds[0], args.seeds[1] + 1)))
    csv_rows: list[dict] = []
    report = build_report(records, repro_dir, csv_rows)
    (repro_dir / "split_comparison.md").write_text(report, encoding="utf-8")
    with open(repro_dir / "split_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_lift(records, fig_dir / "cas_lift_instance_vs_question.png")
    print(f"Wrote {repro_dir / 'split_comparison.md'}, {repro_dir / 'split_comparison.csv'}, "
          f"{fig_dir / 'cas_lift_instance_vs_question.png'}")


if __name__ == "__main__":
    main()
