"""
BCI (Belief-Constrained Inference) — Configuration
Phase 1: Premise Error Analysis on GQA
"""
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path("/data1/DATA/cs22btech11030")
DATA_DIR = DATA_ROOT / "datasets"
GQA_DIR = DATA_DIR / "gqa"
HF_CACHE_DIR = DATA_ROOT / "hf_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ─── GQA Dataset ─────────────────────────────────────────────────────────────
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_SCENE_GRAPHS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

# ─── Experiment Settings ─────────────────────────────────────────────────────
NUM_SAMPLES = 500           # Number of GQA questions for Experiment 1
RANDOM_SEED = 42
MAX_CLAIMS_PER_QUESTION = 10

# ─── VLM Settings ────────────────────────────────────────────────────────────
# LLaVA-1.5 generates only 2 beliefs — too limited. Use Qwen2.5-VL instead.
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_MAX_NEW_TOKENS = 768
VLM_TEMPERATURE = 0.0       # Deterministic for reproducibility
VLM_DEVICE = "cuda:1"        # GPU 1 (most free memory)

# ─── Claim Verification Thresholds ───────────────────────────────────────────
# For scene-graph-based verification
SPATIAL_TOLERANCE = 0.15     # IoU tolerance for spatial relation checks
ATTRIBUTE_SIMILARITY_THRESHOLD = 0.8  # Fuzzy match threshold for attributes

# ─── Error Classification ────────────────────────────────────────────────────
PREMISE_ERROR_THRESHOLD = 0.35  # H1 hypothesis validation threshold
