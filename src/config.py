"""Configuration module for the Bowling Scoreboard CV Extraction system.

Defines project directory paths, default processing parameters, confidence
thresholds, and debug settings.
"""

from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "final output"
SAMPLE_DIR = OUTPUT_DIR / "samples"
DEBUG_DIR = OUTPUT_DIR / "debug"

DEFAULT_VIDEO_PATH = DATA_DIR / "bowling_scoreboard.mp4"
DEFAULT_JSON_PATH = OUTPUT_DIR / "scoreboard_data.json"
DEFAULT_CSV_PATH = OUTPUT_DIR / "scoreboard_summary.csv"
DEFAULT_ANNOTATED_VIDEO_PATH = OUTPUT_DIR / "annotated_bowling_scoreboard.mp4"


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds and parameters for automatic scoreboard detection."""
    min_area_ratio: float = 0.30
    max_area_ratio: float = 0.99
    min_aspect_ratio: float = 1.10
    max_aspect_ratio: float = 2.60
    canny_low: int = 40
    canny_high: int = 140
    min_confidence: float = 0.50


@dataclass(frozen=True)
class LayoutConfig:
    """Parameters for automatic layout and bowling grid geometry."""
    player_count: int = 4
    total_frames: int = 10
    horizontal_kernel_ratio: float = 0.04
    vertical_kernel_ratio: float = 0.03
    max_merge_gap: int = 12


@dataclass(frozen=True)
class TemporalConfig:
    """Parameters for temporal tracking and animation filtering."""
    consensus_frame_count: int = 3
    max_score_jump_allowed: int = 30
    similarity_threshold: float = 0.85