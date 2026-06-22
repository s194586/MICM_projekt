"""Configuration for the local Blobby Volley face controller."""

from pathlib import Path


# Keyboard mapping for Blobby Volley Online.
MOVE_LEFT_KEY = "a"
MOVE_RIGHT_KEY = "d"
JUMP_KEY = "w"
BONUS_KEY = "space"

# Camera settings.
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
MAX_NUM_FACES = 2

# Use one detected face as Player 2 for local jump/bonus testing.
SOLO_TEST_MODE = True
SOLO_TEST_ROLE = "player2"

# Player 1 movement mode. Keep "head_yaw" as the default ergonomic option.
MOVE_CONTROL_MODE = "head_yaw"
# Experimental only. Eye/gaze control is tiring because players must watch the ball.
EXPERIMENTAL_GAZE_MODE = "gaze"

# Detection thresholds. Tune these on the lab camera before the tournament.
HEAD_YAW_LEFT_THRESHOLD = -0.055
HEAD_YAW_RIGHT_THRESHOLD = 0.055
# Hysteresis thresholds use the normalized head_yaw feature, not degrees.
HEAD_YAW_ENTER_THRESHOLD = 0.055
HEAD_YAW_EXIT_THRESHOLD = 0.030
# Player 2 jumps with a rule-based smile detector by default.
JUMP_MODE = "smile"
SMILE_THRESHOLD = 0.45
# Optional fallback for experiments; not used while JUMP_MODE == "smile".
MOUTH_OPEN_THRESHOLD = 0.32
BONUS_PROBA_THRESHOLD = 0.65

# Smoothing / debounce.
ACTION_CONFIRM_FRAMES = 3
JUMP_CONFIRM_FRAMES = 2
BONUS_CONFIRM_FRAMES = 3
JUMP_COOLDOWN_SECONDS = 0.55
BONUS_COOLDOWN_SECONDS = 1.5
KEY_TAP_SECONDS = 0.045

# MediaPipe confidence.
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55

# Paths.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_PATH = DATA_DIR / "gestures.csv"
MODEL_PATH = MODELS_DIR / "bonus_model.pkl"

# Bonus classifier labels.
BONUS_GESTURE_ID = "head_down_nod"
LABEL_NEUTRAL = 0
LABEL_BONUS = 1
LABEL_NAMES = {
    LABEL_NEUTRAL: "neutral",
    LABEL_BONUS: "bonus_gesture",
}
