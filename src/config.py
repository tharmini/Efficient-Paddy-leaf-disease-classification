"""Default hyper-parameters, matching Section IV-B of the paper."""

# Input
IMG_SIZE = (224, 224)
WAVELET_SHAPE = (112, 112, 12)   # Haar DWT of a 224x224x3 image: 4 sub-bands x 3 channels

# Training (Adam, fixed LR, early stopping)
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 7
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42

# Knowledge distillation, Eq. (6): L = (1 - lambda) * CE + lambda * tau^2 * KL
TEMPERATURE = 4.0   # tau
KD_LAMBDA = 0.5     # lambda

# Student: backbone truncated after C5, 256-channel FPN
STUDENT_CHANNELS = 256
STUDENT_TOP_K = 2

# Candidate student branches (code name -> name used in the paper)
ALL_BRANCHES = ["P5_con", "P4_a", "P4_con", "P3_con", "P2_con"]
PAPER_NAMES = {"P5_con": "P5", "P4_a": "P4int", "P4_con": "P4", "P3_con": "P3", "P2_con": "P2"}

# Branches removed by importance-guided structural removal (lowest Eq. (4) scores)
DROP_BRANCHES = ["P2_con", "P4_a"]


def active_branches(drop=DROP_BRANCHES):
    return [b for b in ALL_BRANCHES if b not in drop]


def to_code_names(paper_names):
    """Map paper names such as ['P2', 'P4int'] to code names."""
    inverse = {v.lower(): k for k, v in PAPER_NAMES.items()}
    out = []
    for name in paper_names:
        key = name.strip().lower()
        if key not in inverse:
            raise ValueError(f"Unknown branch '{name}'. Choose from {list(PAPER_NAMES.values())}")
        out.append(inverse[key])
    return out
