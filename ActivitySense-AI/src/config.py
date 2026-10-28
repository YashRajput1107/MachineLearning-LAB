"""Central configuration: paths, labels, feature-group rules and constants."""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (relative to the project root = parent of src/)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "UCI HAR Dataset"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Activities
# ----------------------------------------------------------------------------
ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
DYNAMIC = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
STATIC = ["SITTING", "STANDING", "LAYING"]

# ----------------------------------------------------------------------------
# Feature groups (based on documented UCI HAR feature names)
#   Accelerometer : names containing "Acc", plus the gravity-angle features
#   Gyroscope     : names containing "Gyro"
# ----------------------------------------------------------------------------
ACC_TOKENS = ("Acc", "gravityMean")
GYRO_TOKENS = ("Gyro",)

# reduced feature experiment
SELECTK = 50

# small, interpretable subset for the app's manual-entry model
MANUAL_FEATURES = [
    "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z",
    "tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z",
    "tBodyAccMag-mean()", "tBodyGyroMag-mean()",
]

# ----------------------------------------------------------------------------
# Models and feature-group experiment names
# ----------------------------------------------------------------------------
MODEL_NAMES = ["knn", "svm", "random_forest"]
FEATURE_GROUPS = ["full", "accelerometer", "gyroscope", "reduced"]
