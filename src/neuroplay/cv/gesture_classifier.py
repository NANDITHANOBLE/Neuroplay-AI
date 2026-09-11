"""
Rule-based hand gesture classifier using MediaPipe hand landmarks.
Classifies Rock/Paper/Scissors from 21 hand landmark points via a
finger-extension heuristic (no training required).
"""

from dataclasses import dataclass

from neuroplay.constants import Move
from neuroplay.logger import get_logger

logger = get_logger(__name__)

# MediaPipe hand landmark indices (see MediaPipe Hands documentation)
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class GestureResult:
    move: Move | None
    confidence: float
    fingers_extended: dict[str, bool]


def _is_finger_extended(landmarks, tip_idx: int, mcp_idx: int) -> bool:
    """
    A finger is 'extended' if its tip is farther from the wrist (landmark 0)
    than its base knuckle (MCP joint) — a simple, robust distance heuristic
    that works regardless of hand rotation.
    """
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]

    def dist(a, b) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    return dist(wrist, tip) > dist(wrist, mcp)


def classify_gesture(landmarks) -> GestureResult:
    """
    landmarks: MediaPipe's 21-point hand landmark list (normalized x,y,z per point).
    Returns the classified Move (or None if ambiguous) with a confidence score.
    """
    extended = {
        finger: _is_finger_extended(landmarks, FINGER_TIPS[finger], FINGER_MCPS[finger])
        for finger in FINGER_TIPS
    }

    num_extended = sum(extended.values())
    logger.debug(f"Fingers extended: {extended} | count={num_extended}")

    index_middle_extended = extended["index"] and extended["middle"]
    pinky_curled = not extended["pinky"]

    if num_extended <= 1:
        return GestureResult(move=Move.ROCK, confidence=0.9, fingers_extended=extended)
    elif num_extended >= 4:
        return GestureResult(move=Move.PAPER, confidence=0.9, fingers_extended=extended)
    elif index_middle_extended and pinky_curled and num_extended <= 3:
        # Tolerant of ring-finger ambiguity — only require index+middle clearly
        # extended and pinky clearly curled. Ring finger curl is biomechanically
        # inconsistent across users and sensitive to hand rotation, so we don't
        # require it strictly curled like the original implementation did.
        confidence = 0.85 if num_extended == 2 else 0.65
        return GestureResult(move=Move.SCISSORS, confidence=confidence, fingers_extended=extended)
    else:
        logger.debug(f"Ambiguous gesture: {num_extended} fingers extended, pattern={extended}")
        return GestureResult(move=None, confidence=0.0, fingers_extended=extended)
