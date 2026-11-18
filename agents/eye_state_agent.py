
import numpy as np


class EyeStateAgent:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, close_ratio=0.72):

        self.close_ratio = close_ratio

        self.open_ear_baseline = None
        self.close_threshold = 0.20
        self.open_threshold = 0.23

        self.state = "OPEN"
        self.calibration_samples = []

    @staticmethod
    def calculate_ear(points):

        vertical_1 = np.linalg.norm(
            points[1] - points[5]
        )

        vertical_2 = np.linalg.norm(
            points[2] - points[4]
        )

        horizontal = np.linalg.norm(
            points[0] - points[3]
        )

        if horizontal < 1e-6:
            return 0.0

        return float(
            (vertical_1 + vertical_2) /
            (2.0 * horizontal)
        )

    def calibrate(self, landmarks):

        left_points = landmarks[self.LEFT_EYE, :2]
        right_points = landmarks[self.RIGHT_EYE, :2]

        left_ear = self.calculate_ear(left_points)
        right_ear = self.calculate_ear(right_points)

        average_ear = (left_ear + right_ear) / 2

        # Collect only valid open-eye observations.
        if average_ear > 0.15:
            self.calibration_samples.append(average_ear)

    def finish_calibration(self):

        if len(self.calibration_samples) < 10:
            return False

        self.open_ear_baseline = float(
            np.median(self.calibration_samples)
        )

        self.close_threshold = (
            self.open_ear_baseline * self.close_ratio
        )

        self.open_threshold = (
            self.open_ear_baseline * 0.85
        )

        self.calibration_samples.clear()

        return True

    def process(self, landmarks):

        left_points = landmarks[self.LEFT_EYE, :2]
        right_points = landmarks[self.RIGHT_EYE, :2]

        left_ear = self.calculate_ear(left_points)
        right_ear = self.calculate_ear(right_points)

        average_ear = (left_ear + right_ear) / 2

        # Hysteresis prevents rapid state switching.
        if self.state == "CLOSED":
            if average_ear >= self.open_threshold:
                self.state = "OPEN"
        else:
            if average_ear <= self.close_threshold:
                self.state = "CLOSED"
            elif average_ear < self.open_threshold:
                self.state = "PARTIALLY_CLOSED"
            else:
                self.state = "OPEN"

        # Estimate normalized eye closure risk.
        baseline = self.open_ear_baseline or 0.30

        closed_reference = baseline * 0.45
        denominator = max(
            self.close_threshold - closed_reference,
            0.01
        )

        eye_risk = np.clip(
            (self.close_threshold - average_ear) /
            denominator,
            0.0,
            1.0
        )

        return {
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "average_ear": round(average_ear, 4),
            "eye_state": self.state,
            "eye_closure_risk": round(float(eye_risk), 4),
            "baseline": self.open_ear_baseline
        }
