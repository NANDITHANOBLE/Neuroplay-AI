"""
Standalone demo — opens webcam, displays live gesture classification overlay.
Press 'q' to quit. Useful for manual testing before Streamlit integration (Phase 19).
Run: python -m neuroplay.cv.run_webcam_demo
"""

import cv2

from neuroplay.constants import MOVE_NAMES
from neuroplay.cv.webcam_capture import WebcamGestureDetector
from neuroplay.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    detector = WebcamGestureDetector()
    logger.info("Starting webcam gesture demo. Press 'q' to quit.")

    try:
        while True:
            frame, gesture = detector.read_frame()
            if frame is None:
                break

            if gesture and gesture.move is not None:
                label = f"{MOVE_NAMES[gesture.move]} ({gesture.confidence:.0%})"
                color = (0, 255, 0)
            else:
                label = "No gesture detected"
                color = (0, 0, 255)

            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.imshow("NeuroPlay-AI Webcam Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
        logger.info("Webcam demo stopped.")


if __name__ == "__main__":
    main()
