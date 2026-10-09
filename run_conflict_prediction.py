"""
Run Conflict-Aware Selective Prediction on existing predictions.

This script operates offline — it reads existing response JSONL files,
trains conflict detectors via cross-validation, and evaluates whether
learned conflict scores improve selective prediction over raw confidence.

Two detector variants are compared:
  1. **Within-condition**: supervision = "confidently wrong" (single condition)
  2. **Cross-condition**: supervision = "conflict-susceptible" (paired data)
     The cross-condition detector uses the HealthContradict paired structure:
     for each instance, it compares model confidence under the target condition
     vs. a reference condition (CC).  The label is "would counter-evidence
     change the model's mind?"

Usage
-----
  python run_conflict_prediction.py                          # all models
  python run_conflict_prediction.py --models llama3.1-8b     # one model
  python run_conflict_prediction.py --conditions IC CIC ICC  # subset
  python run_conflict_prediction.py --coverage-targets 1.0 0.75 0.5 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import conflict_detector
import conflict_features
from calibration import auroc_error_detection
from conflict_detector import (
    COVERAGE_TARGETS,
    DetectorResult,
    DetectorMetrics,
    TrainInfo,
    combined_abstention_score,
    confidence_only_score,
    conflict_only_score,
    evaluate_at_exact_coverages,
    evaluate_at_thresholds,
    select_thresholds_at_coverage_targets,
    train_conflict_detector,
    train_cross_condition_detector,
)
from conflict_features import build_cross_condition_labels, build_labels
from data_loader import load_dataset
from selective_prediction import (
    auc_risk_coverage,
    coverage_at_risk,
    risk_coverage_curve,
)
from visualization_conflict import (
    plot_auroc_comparison,
    plot_conflict_risk_coverage,
    plot_conflict_score_distribution,
    plot_conflict_summary_table,
    plot_selective_prediction_combined,
    plot_selective_prediction_conf_only,
)

# NumPy compat
_trapz = getattr(np, "trapezoid", None) or np.trapz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Evaluate abstention signals ──────────────────────────────────────
def _evaluate_signal(
    scores: np.ndarray,
    correct: np.ndarray,
    tau_grid: list[float] | None = None,
) -> dict:
    """Evaluate one abstention signal: AUROC, AUC-RC, coverage@risk."""
    if tau_grid is None:
        tau_grid = [i / 100 for i in range(0, 101)]

    # AUROC for error detection (higher score -> correct)
    auroc = float("nan")
    if len(np.unique(correct)) >= 2:
        auroc = float(roc_auc_score(correct, scores))

    # Risk-coverage
    rc = risk_coverage_curve(scores, correct, tau_grid=tau_grid)
    auc_rc = auc_risk_coverage(rc)
    cov_10 = coverage_at_risk(rc, 0.10)
    cov_5 = coverage_at_risk(rc, 0.05)

    return {
        "auroc": auroc,
        "auc_risk_coverage": auc_rc,
        "coverage_at_10pct_risk": cov_10,
        "coverage_at_5pct_risk": cov_5,
    }


def _evaluate_detector_results(
    results: list[DetectorResult],
    alpha: float,
) -> dict:
    """Evaluate all three abstention signals for a detector's output."""
    conf_scores = confidence_only_score(results)
    confl_scores = conflict_only_score(results)
    comb_scores = combined_abstention_score(results, alpha=alpha)
    correct = np.array([1 - r.is_wrong for r in results])

    return {
        "confidence": _evaluate_signal(conf_scores, correct),
        "conflict_score": _evaluate_signal(confl_scores, correct),
        "combined": _evaluate_signal(comb_scores, correct),
        "_scores": {
            "confidence": conf_scores,
            "conflict": confl_scores,
            "combined": comb_scores,
        },
    }


# ── Per-condition analysis ───────────────────────────────────────────
def run_condition(
    samples: list,
    model_key: str,
    condition: str,
    alpha: float = 0.5,
    reference_condition: str = "CC",
    coverage_targets: list[float] | None = None,
    confidently_wrong_threshold: float = config.HC_CONFIDENTLY_WRONG_THRESHOLD,
    seed: int = config.SEED,
    split_mode: str = "instance",
) -> dict:
    """Train both detector variants and evaluate for one condition."""
    logger.info("=" * 50)
    logger.info("  Condition: %s", condition)
    logger.info("=" * 50)

    correct_for_cond = None  # will be set from first detector

    # ── 1. Within-condition detector ─────────────────────────────────
    logger.info("  --- Within-condition detector ---")
    within_results, within_metrics, within_train_info = train_conflict_detector(
        samples,
        model_key,
        condition,
        build_labels_fn=lambda samples_, model_key_, condition_: build_labels(
            samples_,
            model_key_,
            condition_,
            confidently_wrong_threshold=confidently_wrong_threshold,
        ),
        seed=seed,
        split_mode=split_mode,
    )
    within_eval = _evaluate_detector_results(within_results, alpha)
    correct = np.array([1 - r.is_wrong for r in within_results])
    accuracy = float(correct.mean())

    _log_signal_comparison("Within-condition", within_metrics, within_eval)

    # ── 1b. Selective prediction: threshold selection on train OOF ──
    logger.info("  --- Selective prediction (threshold selection on OOF) ---")
    within_thresholds = select_thresholds_at_coverage_targets(
        within_train_info, alpha=alpha, coverage_targets=coverage_targets,
    )
    within_selective = evaluate_at_thresholds(
        within_results, within_thresholds, alpha=alpha,
    )

    # ── 1c. Selective prediction: exact top-k on held-out test ──────
    logger.info("  --- Selective prediction (exact top-k on test) ---")
    within_selective_exact = evaluate_at_exact_coverages(
        within_results, coverage_targets=coverage_targets, alpha=alpha,
    )

    # ── 2. Cross-condition detector ──────────────────────────────────
    cross_results = None
    cross_metrics = None
    cross_eval = None
    cross_info = None

    # Skip cross-condition for CC (it's the reference itself)
    if condition != reference_condition:
        logger.info("  --- Cross-condition detector (ref=%s) ---", reference_condition)
        cross_results, cross_metrics, cross_info, _cross_train_info = train_cross_condition_detector(
            samples, model_key,
            target_condition=condition,
            reference_condition=reference_condition,
            build_cross_labels_fn=lambda samples_, model_key_, target_condition, reference_condition, delta_threshold: build_cross_condition_labels(
                samples_,
                model_key_,
                target_condition=target_condition,
                reference_condition=reference_condition,
                delta_threshold=delta_threshold,
                confidently_wrong_threshold=confidently_wrong_threshold,
            ),
            seed=seed,
            split_mode=split_mode,
        )
        cross_eval = _evaluate_detector_results(cross_results, alpha)
        _log_signal_comparison("Cross-condition", cross_metrics, cross_eval)
    else:
        logger.info("  Skipping cross-condition for %s (is the reference)", condition)

    # ── Build output dict ────────────────────────────────────────────
    # Convert selective prediction keys to strings for JSON serialization
    selective_json = {
        str(cov_target): metrics
        for cov_target, metrics in within_selective.items()
    }
    selective_exact_json = {
        str(cov_target): metrics
        for cov_target, metrics in within_selective_exact.items()
    }

    output = {
        "condition": condition,
        "accuracy": accuracy,
        "n_test": within_metrics.n_test,
        "n_train": within_metrics.n_train,
        "within_condition": {
            "n_positive_labels": within_metrics.n_confidently_wrong,
            "split_info": within_metrics.split_info,
            "detector_metrics": {
                "auroc_conflict_score": within_metrics.auroc_conflict_score,
                "auroc_confidence": within_metrics.auroc_confidence,
                "mean_conflict_wrong": within_metrics.mean_conflict_wrong,
                "mean_conflict_correct": within_metrics.mean_conflict_correct,
                "fold_aurocs": within_metrics.fold_aurocs,
            },
            "signal_comparison": {
                k: v for k, v in within_eval.items() if not k.startswith("_")
            },
            "selective_prediction": selective_json,
            "selective_prediction_exact_topk": selective_exact_json,
        },
        "_within_results": within_results,
        "_within_scores": within_eval["_scores"],
    }

    if cross_results is not None:
        output["cross_condition"] = {
            "reference": reference_condition,
            "cross_info": cross_info,
            "split_info": cross_metrics.split_info,
            "detector_metrics": {
                "auroc_conflict_score": cross_metrics.auroc_conflict_score,
                "auroc_confidence": cross_metrics.auroc_confidence,
                "mean_conflict_wrong": cross_metrics.mean_conflict_wrong,
                "mean_conflict_correct": cross_metrics.mean_conflict_correct,
                "fold_aurocs": cross_metrics.fold_aurocs,
            },
            "signal_comparison": {
                k: v for k, v in cross_eval.items() if not k.startswith("_")
            },
        }
        output["_cross_results"] = cross_results
        output["_cross_scores"] = cross_eval["_scores"]

    return output


def _log_signal_comparison(variant_name: str, metrics, eval_dict: dict):
    """Log signal comparison for one detector variant."""
    logger.info("  [%s] Detector AUROC (error detect): %.3f",
                variant_name, metrics.auroc_conflict_score)
    for signal_name in ["confidence", "conflict_score", "combined"]:
        if signal_name in eval_dict and not signal_name.startswith("_"):
            m = eval_dict[signal_name]
            logger.info("    %-18s  AUROC=%.3f  AUC-RC=%.3f  Cov@10%%=%.1f%%",
                        signal_name, m["auroc"], m["auc_risk_coverage"],
                        m["coverage_at_10pct_risk"] * 100)


# ── Per-model analysis ───────────────────────────────────────────────
def run_conflict_prediction_for_model(
    model_key: str,
    conditions: list[str] | None = None,
    alpha: float = 0.5,
    reference_condition: str = "CC",
    coverage_targets: list[float] | None = None,
    generate_figures: bool = True,
    confidently_wrong_threshold: float = config.HC_CONFIDENTLY_WRONG_THRESHOLD,
    seed: int = config.SEED,
    split_mode: str = "instance",
) -> dict:
    """Run conflict-aware selective prediction for one model."""
    logger.info("Loading dataset ...")
    samples = load_dataset()

    if conditions is None:
        conditions = ["NC", "CC", "IC", "CIC", "ICC"]

    # Check which conditions have response files
    available = []
    for cond in conditions:
        tid = config.CONDITION_TO_TEMPLATE[cond]
        path = config.RESULTS_DIR / model_key / f"responses_{tid}.jsonl"
        if path.exists():
            available.append(cond)
        else:
            logger.warning("  Skipping %s: response file not found (%s)", cond, path)

    # Also verify reference condition is available
    ref_tid = config.CONDITION_TO_TEMPLATE[reference_condition]
    ref_path = config.RESULTS_DIR / model_key / f"responses_{ref_tid}.jsonl"
    if not ref_path.exists():
        logger.warning("Reference condition %s not available, cross-condition disabled",
                        reference_condition)
        reference_condition = None

    conditions = available

    if not conditions:
        logger.error(
            "No response files found for model=%s in %s. "
            "Run main.py for this model first to generate responses_N.jsonl.",
            model_key,
            config.RESULTS_DIR / model_key,
        )
        return {
            "model": model_key,
            "alpha": alpha,
            "reference_condition": reference_condition,
            "conditions": {},
            "note": (
                "No response files were found for the requested model. "
                "Run main.py first, then rerun conflict prediction."
            ),
        }

    # Run per-condition
    condition_results = {}
    for cond in conditions:
        result = run_condition(
            samples, model_key, cond,
            alpha=alpha,
            reference_condition=reference_condition or cond,
            coverage_targets=coverage_targets,
            confidently_wrong_threshold=confidently_wrong_threshold,
            seed=seed,
            split_mode=split_mode,
        )
        condition_results[cond] = result

    def _noop(*_args, **_kwargs):
        return None

    if generate_figures:
        fig_dir = config.FIGURES_DIR / model_key
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_conflict_score_distribution_fn = plot_conflict_score_distribution
        plot_conflict_risk_coverage_fn = plot_conflict_risk_coverage
        plot_auroc_comparison_fn = plot_auroc_comparison
        plot_conflict_summary_table_fn = plot_conflict_summary_table
        plot_selective_prediction_combined_fn = plot_selective_prediction_combined
        plot_selective_prediction_conf_only_fn = plot_selective_prediction_conf_only
    else:
        fig_dir = config.FIGURES_DIR / model_key
        plot_conflict_score_distribution_fn = _noop
        plot_conflict_risk_coverage_fn = _noop
        plot_auroc_comparison_fn = _noop
        plot_conflict_summary_table_fn = _noop
        plot_selective_prediction_combined_fn = _noop
        plot_selective_prediction_conf_only_fn = _noop

    # ── Generate figures ─────────────────────────────────────────────

    # Per-condition plots — use cross-condition scores where available,
    # fall back to within-condition
    for cond, res in condition_results.items():
        within_results = res["_within_results"]

        # For distribution plots, show cross-condition if available
        if "_cross_results" in res:
            plot_results = res["_cross_results"]
            plot_scores = res["_cross_scores"]
            suffix = "cross"
        else:
            plot_results = within_results
            plot_scores = res["_within_scores"]
            suffix = "within"

        plot_conflict_score_distribution_fn(
            plot_results, cond, model_key, out_dir=fig_dir,
        )
        plot_conflict_risk_coverage_fn(
            plot_results, cond, model_key,
            confidence_scores=plot_scores["confidence"],
            conflict_scores=plot_scores["conflict"],
            combined_scores=plot_scores["combined"],
            out_dir=fig_dir,
        )

    # Cross-condition comparison table — show best of both variants
    condition_metrics_for_plot = {}
    for cond, res in condition_results.items():
        # Use cross-condition if available and better, otherwise within
        within_sc = res["within_condition"]["signal_comparison"]

        if "cross_condition" in res:
            cross_sc = res["cross_condition"]["signal_comparison"]
            # Pick whichever variant has higher combined AUROC
            if cross_sc["combined"]["auroc"] >= within_sc["combined"]["auroc"]:
                best_sc = cross_sc
                best_label = "cross"
            else:
                best_sc = within_sc
                best_label = "within"
        else:
            best_sc = within_sc
            best_label = "within"

        condition_metrics_for_plot[cond] = {
            "accuracy": res["accuracy"],
            "auroc_confidence": best_sc["confidence"]["auroc"],
            "auroc_conflict_score": best_sc["conflict_score"]["auroc"],
            "auroc_combined": best_sc["combined"]["auroc"],
            "auc_rc_confidence": best_sc["confidence"]["auc_risk_coverage"],
            "auc_rc_conflict": best_sc["conflict_score"]["auc_risk_coverage"],
            "auc_rc_combined": best_sc["combined"]["auc_risk_coverage"],
            "cov10_confidence": best_sc["confidence"]["coverage_at_10pct_risk"],
            "cov10_conflict": best_sc["conflict_score"]["coverage_at_10pct_risk"],
            "cov10_combined": best_sc["combined"]["coverage_at_10pct_risk"],
            "best_variant": best_label,
        }

    plot_auroc_comparison_fn(condition_metrics_for_plot, model_key, out_dir=fig_dir)
    plot_conflict_summary_table_fn(
        condition_metrics_for_plot, model_key, out_dir=fig_dir,
    )

    # ── Selective prediction bar charts ──────────────────────────────
    all_selective = {}
    for cond, res in condition_results.items():
        sp = res["within_condition"].get("selective_prediction")
        if sp:
            all_selective[cond] = sp
    if all_selective:
        plot_selective_prediction_combined_fn(
            all_selective, model_key, out_dir=fig_dir,
        )
        plot_selective_prediction_conf_only_fn(
            all_selective, model_key, out_dir=fig_dir,
        )

    # ── Prepare JSON-serializable output ─────────────────────────────
    output = {
        "model": model_key,
        "alpha": alpha,
        "reference_condition": reference_condition,
        "confidently_wrong_threshold": confidently_wrong_threshold,
        "coverage_targets": coverage_targets or COVERAGE_TARGETS,
        "seed": seed,
        "split_mode": split_mode,
        "feature_set": conflict_features.FEATURE_SET,
        "conditions": {},
    }
    for cond, res in condition_results.items():
        out_cond = {k: v for k, v in res.items() if not k.startswith("_")}
        output["conditions"][cond] = out_cond

    return output


# ── Entry point ──────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Conflict-Aware Selective Prediction (offline)"
    )
    p.add_argument(
        "--models", "--model", nargs="+", default=None,
        help="Model keys to analyse (default: auto-detect from results/)",
    )
    p.add_argument(
        "--conditions", nargs="+", default=None,
        help="Conditions to analyse (default: NC CC IC CIC ICC)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for combined score: (1-alpha)*conf - alpha*conflict",
    )
    p.add_argument(
        "--confidently-wrong-threshold",
        type=float,
        default=config.HC_CONFIDENTLY_WRONG_THRESHOLD,
        help="Confidence cutoff for labeling an example as confidently wrong.",
    )
    p.add_argument(
        "--reference", type=str, default="CC",
        help="Reference condition for cross-condition detector (default: CC)",
    )
    p.add_argument(
        "--out-dir", type=str, default=str(config.RESULTS_DIR),
        help="Output directory for JSON results",
    )
    p.add_argument(
        "--coverage-targets", nargs="+", type=float, default=None,
        help=(
            "Coverage targets for selective prediction threshold selection "
            f"(default: {' '.join(str(x) for x in COVERAGE_TARGETS)})"
        ),
    )
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip per-model figure generation. Useful for parameter sweeps.",
    )
    p.add_argument(
        "--split", choices=conflict_detector.SPLIT_MODES, default="instance",
        help=(
            "Train/test split and CV unit: 'instance' (original) or 'question' "
            "(every topic_id wholly in train or wholly in test)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=config.SEED,
        help=f"Seed for the detector split and CV (default: {config.SEED})",
    )
    p.add_argument(
        "--feature-set", choices=conflict_features.FEATURE_SETS, default="full",
        help="Detector feature subset (default: full). Used for ablations.",
    )
    p.add_argument(
        "--embed-cache", type=str, default=None,
        help="Optional .npz file for caching sentence embeddings across runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conflict_features.FEATURE_SET = args.feature_set
    if args.embed_cache:
        conflict_features.EMBED_CACHE_PATH = Path(args.embed_cache)

    # Auto-detect models
    if args.models:
        model_keys = args.models
    else:
        model_keys = [
            d.name for d in config.RESULTS_DIR.iterdir()
            if d.is_dir() and (d / "responses_1.jsonl").exists()
        ]
    if not model_keys:
        logger.error("No models found with response files in %s", config.RESULTS_DIR)
        return

    all_results = {}
    for model_key in model_keys:
        logger.info("=" * 60)
        logger.info("Conflict-Aware Selective Prediction for %s", model_key)
        logger.info("=" * 60)

        result = run_conflict_prediction_for_model(
            model_key,
            conditions=args.conditions,
            alpha=args.alpha,
            reference_condition=args.reference,
            coverage_targets=args.coverage_targets,
            generate_figures=not args.skip_figures,
            confidently_wrong_threshold=args.confidently_wrong_threshold,
            seed=args.seed,
            split_mode=args.split,
        )
        all_results[model_key] = result

        # Save per-model
        result_path = out_dir / f"{model_key}_conflict_prediction.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Results saved -> %s", result_path)

    # Print summary
    _print_summary(all_results)


def _print_summary(all_results: dict) -> None:
    """Print a human-readable summary to stdout."""
    for model_key, result in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"  CONFLICT-AWARE SELECTIVE PREDICTION: {model_key}")
        print(f"  Reference condition: {result.get('reference_condition', 'CC')}")
        print(f"  Confidently-wrong threshold: {result.get('confidently_wrong_threshold', config.HC_CONFIDENTLY_WRONG_THRESHOLD):.2f}")
        print(f"{'=' * 80}")

        for cond, cond_result in result["conditions"].items():
            acc = cond_result["accuracy"]

            n_tr = cond_result.get("n_train", "?")
            n_te = cond_result.get("n_test", "?")
            print(f"\n  Condition: {cond}  (accuracy={acc:.3f}, train={n_tr}, test={n_te})")

            # Within-condition
            wc = cond_result["within_condition"]
            wsc = wc["signal_comparison"]
            print(f"    [Within-condition]  label=confidently_wrong  n_pos={wc['n_positive_labels']} (in test set)")
            _print_signal_table(wsc)

            # Selective prediction table
            if "selective_prediction" in wc:
                _print_selective_prediction_table(
                    wc["selective_prediction"],
                    acc,
                    title="Selective Prediction — train-threshold transfer",
                )
            if "selective_prediction_exact_topk" in wc:
                _print_selective_prediction_table(
                    wc["selective_prediction_exact_topk"],
                    acc,
                    title="Selective Prediction — exact top-k on test",
                )

            # Cross-condition
            if "cross_condition" in cond_result:
                xc = cond_result["cross_condition"]
                xsc = xc["signal_comparison"]
                xi = xc["cross_info"]
                print(f"    [Cross-condition]  label={xi['label_type']}  "
                      f"ref={xi['reference_condition']}  "
                      f"n_susceptible={xi['n_conflict_susceptible']}  "
                      f"n_flips={xi['n_prediction_flips']}  "
                      f"mean_delta={xi['mean_confidence_delta']:.3f}")
                _print_signal_table(xsc)


def _print_signal_table(sc: dict) -> None:
    """Print a formatted signal comparison sub-table."""
    print(f"      {'Signal':<20s}  {'AUROC':>8s}  {'AUC-RC':>8s}  {'Cov@10%':>8s}  {'Cov@5%':>8s}")
    print(f"      {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for signal_name, metrics in sc.items():
        auroc = metrics.get("auroc", float("nan"))
        auc_rc = metrics.get("auc_risk_coverage", float("nan"))
        cov10 = metrics.get("coverage_at_10pct_risk", 0)
        cov5 = metrics.get("coverage_at_5pct_risk", 0)
        print(f"      {signal_name:<20s}  {auroc:>8.3f}  {auc_rc:>8.3f}  {cov10:>7.1%}  {cov5:>7.1%}")


def _print_selective_prediction_table(
    sp: dict,
    baseline_acc: float,
    title: str,
) -> None:
    """Print one selective prediction table."""
    print()
    print(f"    [{title}]")
    print(f"    Baseline accuracy (answer all): {baseline_acc:.1%}")
    print(f"      {'Cov Target':>10s}  {'Acc(comb)':>10s}  {'Cov(comb)':>10s}  "
          f"{'Acc(conf)':>10s}  {'Cov(conf)':>10s}  {'Lift(comb)':>10s}  {'Lift(conf)':>10s}")
    print(f"      {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for cov_target_str, m in sorted(sp.items(), key=lambda x: -float(x[0])):
        cov_t = float(cov_target_str)
        acc_c = m["test_accuracy_combined"]
        cov_c = m["test_coverage_combined"]
        acc_f = m["test_accuracy_conf_only"]
        cov_f = m["test_coverage_conf_only"]
        lift_c = m["accuracy_lift_combined"]
        lift_f = m["accuracy_lift_conf_only"]
        print(f"      {cov_t:>9.0%}  {acc_c:>10.1%}  {cov_c:>10.1%}  "
              f"{acc_f:>10.1%}  {cov_f:>10.1%}  {lift_c:>+10.1%}  {lift_f:>+10.1%}")


if __name__ == "__main__":
    main()
