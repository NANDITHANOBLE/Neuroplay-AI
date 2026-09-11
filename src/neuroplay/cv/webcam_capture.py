"""
Webcam capture + MediaPipe hand detection pipeline.
Provides a simple interface: capture a frame, detect hand landmarks,
classify the gesture, return the result (with the annotated frame for UI display).
"""

import cv2
import mediapipe as mp
import numpy as np

from neuroplay.cv.gesture_classifier import GestureResult, classify_gesture
from neuroplay.logger import get_logger

logger = get_logger(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class WebcamGestureDetector:
    """Wraps MediaPipe Hands + OpenCV webcam capture for real-time gesture detection."""

    def __init__(self, camera_index: int = 0, min_detection_confidence: float = 0.7):
        self.cap = cv2.VideoCapture(camera_index)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def read_frame(self) -> tuple[np.ndarray | None, GestureResult | None]:
        """
        Captures one frame, detects hand landmarks, classifies gesture.
        Returns (annotated_frame, gesture_result). Both None if capture fails.
        """
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to read frame from webcam.")
            return None, None

        frame = cv2.flip(frame, 1)  # mirror for natural user-facing view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_result = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_result = classify_gesture(hand_landmarks.landmark)

        return frame, gesture_result

    def release(self) -> None:
        self.cap.release()
        self.hands.close()
