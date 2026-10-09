"""
Configuration for Uncertainty-Aware Evaluation of Biomedical LLMs.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
HC_DIR = PROJECT_ROOT / "HealthContradict-main"
DATASET_JSONL = HC_DIR / "dataset" / "dataset_ready.jsonl"
PROMPTS_DIR = HC_DIR / "dataset"           # prompts_1.jsonl … prompts_5.jsonl
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Models ─────────────────────────────────────────────────────────────
MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "meditron3-8b": "OpenMeditron/Meditron3-8B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "phi4-14b": "microsoft/phi-4",
}

CHAT_TEMPLATE_KWARGS = {
    "qwen3-8b": {"enable_thinking": False},
    "qwen3.5-9b": {"enable_thinking": False},
}

# ── Context conditions ──────────────────────────────────────────────────
# Mapping: condition name → template_id in HealthContradict prompts
CONDITIONS = ["NC", "CC", "IC", "CIC", "ICC"]
CONDITION_TO_TEMPLATE = {
    "NC": 1,    # parametric only
    "CC": 2,    # correct context only
    "IC": 3,    # incorrect context only
    "CIC": 4,   # correct-first, then incorrect
    "ICC": 5,   # incorrect-first, then correct
}
TEMPLATE_TO_CONDITION = {v: k for k, v in CONDITION_TO_TEMPLATE.items()}

# ── Answer tokens ───────────────────────────────────────────────────────
# The HealthContradict repo uses uppercase "YES" / "NO" for token lookup.
# We also include lowercase/title-case variants for robustness.
YES_TOKENS = ["YES", "Yes", "yes"]
NO_TOKENS = ["NO", "No", "no"]

# ── Calibration ─────────────────────────────────────────────────────────
ECE_NUM_BINS = 10

# ── Selective prediction ────────────────────────────────────────────────
TAU_GRID = [i / 100 for i in range(50, 100)]  # confidence thresholds 0.50 … 0.99

# ── Conflict detector ───────────────────────────────────────────────────
# Referenced by conflict_features.py / run_conflict_prediction.py but missing
# from the released config. Values match the paper (tau = 0.7) and the
# delta_threshold recorded in the cached results/*_conflict_prediction.json.
HC_CONFIDENTLY_WRONG_THRESHOLD = 0.7
HC_CROSS_CONDITION_DELTA_THRESHOLD = 0.3

# ── Inference ───────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1  # constrained to single Yes/No token
DEVICE = "cuda"     # set to "cpu" if no GPU available
DTYPE = "float16"   # HealthContradict repo uses float16

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42
