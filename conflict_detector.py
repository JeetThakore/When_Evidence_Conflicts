"""
Conflict-Aware Selective Prediction -- Detector Training.

Two detector variants:

1. **Within-condition** (primary): predicts "confidently wrong" using features
   from a single condition. The exact confidence threshold is dataset-specific.

2. **Cross-condition** (ablation): predicts "conflict-susceptible" using the
   paired data structure of HealthContradict.

Evaluation protocol:
  - A stratified 80/20 train/test split is performed first.
  - 5-fold CV runs on the 80% train split for validation.
  - A final model is trained on the full 80% train split.
  - All reported metrics come from the held-out 20% test set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config
from data_loader import HealthContradictSample
from conflict_features import (
    build_features,
    build_labels,
    build_cross_condition_labels,
    load_uncertainty_signals,
)

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2

# "instance": instance-level split and CV (original behaviour).
# "question": every topic_id goes wholly into train or wholly into test,
#             in both the 80/20 split and the CV on the train part.
SPLIT_MODES = ("instance", "question")
SPLIT_MODE = "instance"


@dataclass
class DetectorResult:
    conflict_score: float
    model_confidence: float
    is_wrong: int
    confidently_wrong: int
    ground_truth: str
    predicted_label: str


@dataclass
class DetectorMetrics:
    auroc_conflict_score: float
    auroc_confidence: float
    mean_conflict_wrong: float
    mean_conflict_correct: float
    fold_aurocs: list[float]
    n_confidently_wrong: int
    n_total: int
    n_train: int
    n_test: int
    split_info: dict | None = None


@dataclass
class TrainInfo:
    """Out-of-fold scores and labels from the training split.

    Used for threshold selection: the OOF conflict scores are honest
    (each instance scored by a model that didn't see it during training).
    """
    oof_conflict_scores: np.ndarray   # shape (n_train,)
    confidence: np.ndarray            # shape (n_train,)  — raw model confidence
    is_wrong: np.ndarray              # shape (n_train,)  — 1=incorrect, 0=correct


def _describe_label_distribution(y: np.ndarray) -> str:
    classes, counts = np.unique(y, return_counts=True)
    return ", ".join(f"{int(cls)}:{int(cnt)}" for cls, cnt in zip(classes, counts))


def _resolve_split_mode(split_mode):
    split_mode = SPLIT_MODE if split_mode is None else split_mode
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
    return split_mode


def _question_train_test_split(all_idx, y, groups, test_size, seed, label_name="target"):
    """Group-level split: no topic_id appears in both train and test.

    Uses one fold of StratifiedGroupKFold (n_splits = round(1/test_size)) as
    the test set, falling back to GroupShuffleSplit if stratification fails.
    """
    n_splits = int(round(1.0 / test_size))
    train_idx = test_idx = None
    method = "StratifiedGroupKFold"
    if len(np.unique(y)) >= 2:
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            tr, te = next(sgkf.split(all_idx, y, groups))
            if len(np.unique(y[te])) >= 2 and len(np.unique(y[tr])) >= 2:
                train_idx, test_idx = all_idx[tr], all_idx[te]
        except ValueError as exc:
            logger.warning("  StratifiedGroupKFold failed for %s: %s", label_name, exc)
    if train_idx is None:
        method = "GroupShuffleSplit"
        logger.warning(
            "  %s: stratified group split not possible (%s); "
            "falling back to GroupShuffleSplit.",
            label_name, _describe_label_distribution(y),
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(all_idx, y, groups))
        train_idx, test_idx = all_idx[tr], all_idx[te]

    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    assert not (train_groups & test_groups), "topic_id leaked between train and test"
    logger.info(
        "  Question split [%s, %s]: test fraction %.3f (%d/%d instances), "
        "%d test questions, %d train questions",
        label_name, method, len(test_idx) / len(all_idx), len(test_idx),
        len(all_idx), len(test_groups), len(train_groups),
    )
    return train_idx, test_idx, method


def _split_info(split_mode, method, groups, train_idx, test_idx, y):
    tr_g, te_g = set(groups[train_idx].tolist()), set(groups[test_idx].tolist())
    return {
        "split_mode": split_mode,
        "split_method": method,
        "test_fraction": len(test_idx) / (len(train_idx) + len(test_idx)),
        "n_train_positive_labels": int(y[train_idx].sum()),
        "n_test_positive_labels": int(y[test_idx].sum()),
        "n_train_questions": len(tr_g),
        "n_test_questions": len(te_g),
        "n_shared_questions": len(tr_g & te_g),
    }


def _safe_train_test_split(all_idx, y, test_size, seed, label_name="target"):
    classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) >= 2 and counts.min() >= 2
    if can_stratify:
        return train_test_split(
            all_idx, test_size=test_size, stratify=y, random_state=seed,
        )

    logger.warning(
        "  %s has insufficient class support for stratified split (%s); "
        "falling back to unstratified split.",
        label_name, _describe_label_distribution(y),
    )
    return train_test_split(all_idx, test_size=test_size, random_state=seed)


def _cv_on_train(X_train, y_train, is_wrong_train, n_folds, seed, label_name="target",
                 groups_train=None):
    """5-fold CV on train split.  Returns fold AUROCs *and* OOF conflict scores.

    If ``groups_train`` is given, folds are built with StratifiedGroupKFold so
    that no group appears in both the fitting and the validation part.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        constant_score = float(classes[0]) if len(classes) else 0.0
        logger.warning(
            "  CV skipped for %s: training labels contain a single class (%s). "
            "Using constant conflict score %.1f.",
            label_name, _describe_label_distribution(y_train), constant_score,
        )
        return [], np.full(len(y_train), constant_score, dtype=float)

    n_splits = min(n_folds, int(counts.min()))
    if groups_train is not None:
        n_splits = min(n_splits, len(np.unique(groups_train)))
    if n_splits < 2:
        constant_score = float(classes[-1])
        logger.warning(
            "  CV skipped for %s: minority class too small for stratified folds (%s). "
            "Using constant conflict score %.1f.",
            label_name, _describe_label_distribution(y_train), constant_score,
        )
        return [], np.full(len(y_train), constant_score, dtype=float)

    if n_splits != n_folds:
        logger.warning(
            "  Reducing CV folds for %s from %d to %d due to label imbalance (%s).",
            label_name, n_folds, n_splits, _describe_label_distribution(y_train),
        )

    fold_aurocs = []
    oof_scores = np.full(len(y_train), np.nan)
    if groups_train is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = skf.split(X_train, y_train)
    else:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = skf.split(X_train, y_train, groups_train)
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_iter):
        if groups_train is not None:
            assert not (set(groups_train[tr_idx].tolist())
                        & set(groups_train[val_idx].tolist())), \
                "topic_id leaked between CV folds"
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        if groups_train is not None and len(np.unique(y_tr)) < 2:
            # Group folds cannot always keep both classes in the fitting part
            # when positives are rare; score the fold with a constant.
            logger.warning(
                "  Fold %d [%s]: fitting part has a single class (%s); "
                "using constant conflict score %.1f.",
                fold_idx, label_name, _describe_label_distribution(y_tr), float(y_tr[0]),
            )
            oof_scores[val_idx] = float(y_tr[0])
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=seed, class_weight="balanced",
        )
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_val_s)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
        scores = proba[:, pos_idx] if pos_idx >= 0 else np.zeros(len(val_idx))
        oof_scores[val_idx] = scores
        is_wrong_val = is_wrong_train[val_idx]
        if len(np.unique(is_wrong_val)) >= 2:
            fold_auc = roc_auc_score(is_wrong_val, scores)
            fold_aurocs.append(fold_auc)
            logger.info("  Fold %d [%s]: AUROC=%.3f (n_val=%d)",
                        fold_idx, label_name, fold_auc, len(val_idx))
        else:
            logger.info("  Fold %d [%s]: skipped AUROC (single class)",
                        fold_idx, label_name)
    return fold_aurocs, oof_scores


def _train_final_and_predict(X_train, y_train, X_test, seed):
    classes = np.unique(y_train)
    if len(classes) < 2:
        constant_score = float(classes[0]) if len(classes) else 0.0
        logger.warning(
            "  Final detector skipped: training labels contain a single class (%s). "
            "Using constant conflict score %.1f on test set.",
            _describe_label_distribution(y_train), constant_score,
        )
        return np.full(X_test.shape[0], constant_score, dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        random_state=seed, class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)
    pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
    if pos_idx >= 0:
        return proba[:, pos_idx]
    return np.zeros(X_test.shape[0])


def _build_test_results_and_metrics(
    conflict_scores, fold_aurocs, is_wrong, confidence,
    y_label, predicted_labels, samples, test_idx, n_train,
):
    ground_truths = [samples[i].answer for i in test_idx]
    n_test = len(test_idx)
    results = [
        DetectorResult(
            conflict_score=float(conflict_scores[i]),
            model_confidence=float(confidence[i]),
            is_wrong=int(is_wrong[i]),
            confidently_wrong=int(y_label[i]),
            ground_truth=ground_truths[i],
            predicted_label=str(predicted_labels[i]),
        )
        for i in range(n_test)
    ]
    correct_mask = is_wrong == 0
    wrong_mask = is_wrong == 1
    auroc_cs = float("nan")
    if len(np.unique(is_wrong)) >= 2:
        auroc_cs = float(roc_auc_score(is_wrong, conflict_scores))
    auroc_conf = float("nan")
    if len(np.unique(is_wrong)) >= 2:
        auroc_conf = float(roc_auc_score(1 - is_wrong, confidence))
    metrics = DetectorMetrics(
        auroc_conflict_score=auroc_cs,
        auroc_confidence=auroc_conf,
        mean_conflict_wrong=float(conflict_scores[wrong_mask].mean()) if wrong_mask.sum() > 0 else 0.0,
        mean_conflict_correct=float(conflict_scores[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0,
        fold_aurocs=fold_aurocs,
        n_confidently_wrong=int(y_label.sum()),
        n_total=n_test,
        n_train=n_train,
        n_test=n_test,
    )
    return results, metrics


def train_conflict_detector(
    samples, model_key, condition,
    n_folds=5, test_size=TEST_SIZE, seed=config.SEED,
    build_features_fn=None, build_labels_fn=None, load_signals_fn=None,
    split_mode=None,
):
    """Train within-condition conflict detector with train/test split.

    1. Build features and labels for all instances.
    2. Stratified 80/20 split on the supervision label.
    3. 5-fold CV on train split (for validation AUROCs).
    4. Train final model on full train split.
    5. Predict on held-out test split.
    6. Report all metrics on test set only.

    The ``build_features_fn``, ``build_labels_fn``, and ``load_signals_fn``
    parameters allow injecting dataset-specific implementations (e.g. for
    ConflictBank) while keeping the training logic unchanged.
    """
    if build_features_fn is None:
        build_features_fn = build_features
    if build_labels_fn is None:
        build_labels_fn = build_labels
    if load_signals_fn is None:
        load_signals_fn = load_uncertainty_signals

    logger.info("Building features for condition=%s (within-condition) ...", condition)
    X = build_features_fn(samples, model_key, condition)
    labels = build_labels_fn(samples, model_key, condition)
    y = labels["confidently_wrong"]
    is_wrong = labels["is_wrong"]
    confidence = labels["confidence"]
    template_id = config.CONDITION_TO_TEMPLATE[condition]
    unc = load_signals_fn(model_key, template_id)
    predicted_labels = unc["predicted_label"]

    all_idx = np.arange(len(samples))
    split_mode = _resolve_split_mode(split_mode)
    groups = np.array([s.topic_id for s in samples])
    if split_mode == "question":
        train_idx, test_idx, split_method = _question_train_test_split(
            all_idx, y, groups, test_size=test_size, seed=seed,
            label_name="confidently_wrong",
        )
    else:
        train_idx, test_idx = _safe_train_test_split(
            all_idx, y, test_size=test_size, seed=seed,
            label_name="confidently_wrong",
        )
        split_method = "train_test_split"
    logger.info("  Train/test split: %d train, %d test", len(train_idx), len(test_idx))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]
    is_wrong_train = is_wrong[train_idx]

    fold_aurocs, oof_scores = _cv_on_train(
        X_train, y_train, is_wrong_train, n_folds, seed,
        label_name="confidently_wrong",
        groups_train=groups[train_idx] if split_mode == "question" else None,
    )
    test_scores = _train_final_and_predict(X_train, y_train, X_test, seed)

    results, metrics = _build_test_results_and_metrics(
        test_scores, fold_aurocs,
        is_wrong[test_idx], confidence[test_idx],
        y[test_idx], predicted_labels[test_idx],
        samples, test_idx, n_train=len(train_idx),
    )
    metrics.split_info = _split_info(split_mode, split_method, groups, train_idx, test_idx, y)

    train_info = TrainInfo(
        oof_conflict_scores=oof_scores,
        confidence=confidence[train_idx],
        is_wrong=is_wrong[train_idx],
    )
    return results, metrics, train_info


def train_cross_condition_detector(
    samples, model_key, target_condition,
    reference_condition="CC", label_type="conflict_susceptible",
    delta_threshold=0.3, n_folds=5, test_size=TEST_SIZE, seed=config.SEED,
    build_features_fn=None, build_cross_labels_fn=None, load_signals_fn=None,
    split_mode=None,
):
    """Train cross-condition conflict detector with train/test split."""
    if build_features_fn is None:
        build_features_fn = build_features
    if build_cross_labels_fn is None:
        build_cross_labels_fn = build_cross_condition_labels
    if load_signals_fn is None:
        load_signals_fn = load_uncertainty_signals

    logger.info(
        "Building features for %s (cross-condition, ref=%s, label=%s) ...",
        target_condition, reference_condition, label_type,
    )
    X = build_features_fn(samples, model_key, target_condition)
    cross_labels = build_cross_labels_fn(
        samples, model_key,
        target_condition=target_condition,
        reference_condition=reference_condition,
        delta_threshold=delta_threshold,
    )
    y = cross_labels[label_type]
    is_wrong = cross_labels["is_wrong"]
    confidence = cross_labels["confidence"]
    template_id = config.CONDITION_TO_TEMPLATE[target_condition]
    unc = load_signals_fn(model_key, template_id)
    predicted_labels = unc["predicted_label"]

    logger.info("  Label distribution: %d positive / %d total (%.1f%%)",
                y.sum(), len(y), 100 * y.mean())

    all_idx = np.arange(len(samples))
    split_mode = _resolve_split_mode(split_mode)
    groups = np.array([s.topic_id for s in samples])
    if split_mode == "question":
        train_idx, test_idx, split_method = _question_train_test_split(
            all_idx, y, groups, test_size=test_size, seed=seed,
            label_name=label_type,
        )
    else:
        train_idx, test_idx = _safe_train_test_split(
            all_idx, y, test_size=test_size, seed=seed,
            label_name=label_type,
        )
        split_method = "train_test_split"
    logger.info("  Train/test split: %d train, %d test", len(train_idx), len(test_idx))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]
    is_wrong_train = is_wrong[train_idx]

    fold_aurocs, oof_scores = _cv_on_train(
        X_train, y_train, is_wrong_train, n_folds, seed,
        label_name=label_type,
        groups_train=groups[train_idx] if split_mode == "question" else None,
    )
    test_scores = _train_final_and_predict(X_train, y_train, X_test, seed)

    results, metrics = _build_test_results_and_metrics(
        test_scores, fold_aurocs,
        is_wrong[test_idx], confidence[test_idx],
        y[test_idx], predicted_labels[test_idx],
        samples, test_idx, n_train=len(train_idx),
    )
    metrics.split_info = _split_info(split_mode, split_method, groups, train_idx, test_idx, y)

    train_info = TrainInfo(
        oof_conflict_scores=oof_scores,
        confidence=confidence[train_idx],
        is_wrong=is_wrong[train_idx],
    )

    cross_info = {
        "target_condition": target_condition,
        "reference_condition": reference_condition,
        "label_type": label_type,
        "delta_threshold": delta_threshold,
        "n_conflict_susceptible": int(cross_labels["conflict_susceptible"].sum()),
        "n_confident_and_flips": int(cross_labels["confident_and_flips"].sum()),
        "n_prediction_flips": int(cross_labels["prediction_flips"].sum()),
        "mean_confidence_delta": float(cross_labels["confidence_delta"].mean()),
        "mean_logit_margin_delta": float(cross_labels["logit_margin_delta"].mean()),
    }
    return results, metrics, cross_info, train_info


def combined_abstention_score(results, alpha=0.5):
    """Combined: (1-alpha)*confidence - alpha*conflict_score. Higher=trustworthy."""
    confs = np.array([r.model_confidence for r in results])
    conflicts = np.array([r.conflict_score for r in results])
    return (1.0 - alpha) * confs - alpha * conflicts


def conflict_only_score(results):
    """1 - conflict_score as abstention signal (higher = more trustworthy)."""
    return 1.0 - np.array([r.conflict_score for r in results])


def confidence_only_score(results):
    """Baseline: raw model confidence as abstention signal."""
    return np.array([r.model_confidence for r in results])


# ── Threshold selection on train OOF scores ─────────────────────────


COVERAGE_TARGETS = [1.0, 0.75, 0.5, 0.25]


def _compute_combined_from_raw(confidence, conflict_scores, alpha=0.5):
    """Compute combined abstention score from raw arrays."""
    return (1.0 - alpha) * confidence - alpha * conflict_scores


def _select_top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a deterministic mask for the top-k scores."""
    n = len(scores)
    k = max(0, min(k, n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    if k == n:
        mask[:] = True
        return mask

    # Stable descending sort gives deterministic tie-breaking.
    order = np.argsort(-scores, kind="mergesort")
    mask[order[:k]] = True
    return mask


def select_thresholds_at_coverage_targets(
    train_info: TrainInfo,
    alpha: float = 0.5,
    coverage_targets: list[float] | None = None,
) -> dict[float, dict]:
    """Select abstention thresholds on the train OOF scores for each coverage target.

    For each target coverage level, finds the threshold that achieves at least
    that coverage while maximizing accuracy among the answered instances.

    Parameters
    ----------
    train_info : TrainInfo
        OOF conflict scores, confidence, and is_wrong for training instances.
    alpha : float
        Weight for combined score.
    coverage_targets : list[float]
        Desired coverage levels (e.g., [1.0, 0.75, 0.5, 0.25]).

    Returns
    -------
    dict mapping coverage_target -> {threshold, train_accuracy, train_coverage}
    """
    if coverage_targets is None:
        coverage_targets = COVERAGE_TARGETS
    coverage_targets = sorted(coverage_targets, reverse=True)

    for cov_target in coverage_targets:
        if cov_target <= 0.0 or cov_target > 1.0:
            raise ValueError(
                f"Coverage targets must lie in (0, 1], got {cov_target}."
            )

    oof_combined = _compute_combined_from_raw(
        train_info.confidence, train_info.oof_conflict_scores, alpha,
    )
    is_correct = 1 - train_info.is_wrong
    n = len(oof_combined)

    # Sort scores to sweep thresholds efficiently
    sorted_scores = np.sort(oof_combined)

    results = {}
    for cov_target in coverage_targets:
        # The threshold is the score at the (1 - cov_target) quantile:
        # we keep the top cov_target fraction, so threshold = percentile((1-cov)*100)
        quantile_idx = int(np.ceil((1.0 - cov_target) * n)) - 1
        quantile_idx = max(0, min(quantile_idx, n - 1))
        tau = sorted_scores[quantile_idx]

        mask = oof_combined >= tau
        actual_coverage = mask.sum() / n
        accuracy = float(is_correct[mask].mean()) if mask.sum() > 0 else 0.0

        results[cov_target] = {
            "threshold": float(tau),
            "train_accuracy": accuracy,
            "train_coverage": actual_coverage,
            "train_n_answered": int(mask.sum()),
        }
        logger.info(
            "  OOF threshold selection: cov_target=%.0f%%  tau=%.4f  "
            "train_acc=%.3f  train_cov=%.1f%%  n_answered=%d/%d",
            cov_target * 100, tau, accuracy, actual_coverage * 100,
            mask.sum(), n,
        )

    return results


def evaluate_at_thresholds(
    test_results: list[DetectorResult],
    thresholds: dict[float, dict],
    alpha: float = 0.5,
) -> dict[float, dict]:
    """Apply train-selected thresholds to test set, report accuracy and coverage.

    Parameters
    ----------
    test_results : list[DetectorResult]
        Detector results on the held-out test set.
    thresholds : dict
        Output of select_thresholds_at_coverage_targets.
    alpha : float
        Weight for combined score.

    Returns
    -------
    dict mapping coverage_target -> {threshold, test_accuracy, test_coverage, ...}
    """
    combined = combined_abstention_score(test_results, alpha)
    is_correct = np.array([1 - r.is_wrong for r in test_results])
    confidence = np.array([r.model_confidence for r in test_results])
    n = len(test_results)
    baseline_accuracy = float(is_correct.mean())

    results = {}
    for cov_target, info in thresholds.items():
        tau = info["threshold"]

        # Combined signal
        mask_combined = combined >= tau
        cov_combined = mask_combined.sum() / n
        acc_combined = (
            float(is_correct[mask_combined].mean())
            if mask_combined.sum() > 0 else 0.0
        )

        # Confidence-only at same coverage: pick the tau for confidence
        # that gives the same number of answered instances
        n_answer = mask_combined.sum()
        if n_answer > 0 and n_answer < n:
            conf_threshold = np.sort(confidence)[-(n_answer)]
            mask_conf = confidence >= conf_threshold
        elif n_answer >= n:
            mask_conf = np.ones(n, dtype=bool)
        else:
            mask_conf = np.zeros(n, dtype=bool)
        acc_conf_at_same_cov = (
            float(is_correct[mask_conf].mean())
            if mask_conf.sum() > 0 else 0.0
        )
        cov_conf_at_same_cov = mask_conf.sum() / n

        results[cov_target] = {
            "threshold": float(tau),
            "test_accuracy_combined": acc_combined,
            "test_coverage_combined": cov_combined,
            "test_n_answered_combined": int(mask_combined.sum()),
            "test_accuracy_conf_only": acc_conf_at_same_cov,
            "test_coverage_conf_only": cov_conf_at_same_cov,
            "test_n_answered_conf_only": int(mask_conf.sum()),
            "baseline_accuracy": baseline_accuracy,
            "accuracy_lift_combined": acc_combined - baseline_accuracy,
            "accuracy_lift_conf_only": acc_conf_at_same_cov - baseline_accuracy,
        }

    return results


def evaluate_at_exact_coverages(
    test_results: list[DetectorResult],
    coverage_targets: list[float] | None = None,
    alpha: float = 0.5,
) -> dict[float, dict]:
    """Evaluate exact top-k selective prediction on the held-out test set.

    Unlike threshold transfer, this protocol does not deploy a numeric
    threshold chosen on train. For each target coverage, it keeps exactly the
    top-k test examples ranked by score, where k is determined by the desired
    coverage level.
    """
    if coverage_targets is None:
        coverage_targets = COVERAGE_TARGETS
    coverage_targets = sorted(coverage_targets, reverse=True)

    for cov_target in coverage_targets:
        if cov_target <= 0.0 or cov_target > 1.0:
            raise ValueError(
                f"Coverage targets must lie in (0, 1], got {cov_target}."
            )

    combined = combined_abstention_score(test_results, alpha)
    confidence = confidence_only_score(test_results)
    is_correct = np.array([1 - r.is_wrong for r in test_results])
    n = len(test_results)
    baseline_accuracy = float(is_correct.mean())

    results = {}
    for cov_target in coverage_targets:
        n_answer = int(round(cov_target * n))
        if cov_target > 0.0:
            n_answer = max(1, n_answer)
        n_answer = min(n_answer, n)

        mask_combined = _select_top_k_mask(combined, n_answer)
        mask_conf = _select_top_k_mask(confidence, n_answer)

        cov_combined = mask_combined.sum() / n if n > 0 else 0.0
        cov_conf = mask_conf.sum() / n if n > 0 else 0.0
        acc_combined = (
            float(is_correct[mask_combined].mean())
            if mask_combined.sum() > 0 else 0.0
        )
        acc_conf = (
            float(is_correct[mask_conf].mean())
            if mask_conf.sum() > 0 else 0.0
        )

        results[cov_target] = {
            "selection_mode": "exact_top_k",
            "target_n_answered": int(n_answer),
            "test_accuracy_combined": acc_combined,
            "test_coverage_combined": cov_combined,
            "test_n_answered_combined": int(mask_combined.sum()),
            "test_accuracy_conf_only": acc_conf,
            "test_coverage_conf_only": cov_conf,
            "test_n_answered_conf_only": int(mask_conf.sum()),
            "baseline_accuracy": baseline_accuracy,
            "accuracy_lift_combined": acc_combined - baseline_accuracy,
            "accuracy_lift_conf_only": acc_conf - baseline_accuracy,
        }

    return results
