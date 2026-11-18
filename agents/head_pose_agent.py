
import cv2
import numpy as np
import time


class HeadPoseAgent:

    # MediaPipe Face Mesh landmark indices
    LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

    # Generic 3D reference face model
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),            # Nose
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye
        (225.0, 170.0, -135.0),     # Right eye
        (-150.0, -150.0, -125.0),   # Left mouth
        (150.0, -150.0, -125.0)     # Right mouth
    ], dtype=np.float64)

    def __init__(
        self,
        smoothing_alpha=0.25,
        yaw_threshold=20.0,
        pitch_threshold=15.0,
        distraction_delay=1.5
    ):

        self.smoothing_alpha = smoothing_alpha

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        self.distraction_delay = distraction_delay

        self.camera_matrix = None

        self.dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        # Calibration
        self.neutral_pose = None
        self.calibration_samples = []

        # Smoothed angles
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None

        # Temporal distraction tracking
        self.distraction_start = None
        self.distraction_duration = 0.0

        self.last_timestamp = None

    # ======================================
    # HELPER FUNCTIONS
    # ======================================

    @staticmethod
    def clamp(value):

        return float(
            np.clip(value, 0.0, 1.0)
        )

    @staticmethod
    def normalize_angle(angle):

        return (
            (angle + 180.0) % 360.0
        ) - 180.0

    def smooth_angle(self, previous, current):

        if previous is None:
            return current

        difference = self.normalize_angle(
            current - previous
        )

        return self.normalize_angle(
            previous +
            self.smoothing_alpha * difference
        )

    # ======================================
    # CALIBRATION
    # ======================================

    def calibrate(self, yaw, pitch, roll):

        self.calibration_samples.append([
            yaw,
            pitch,
            roll
        ])

    def finish_calibration(self):

        if len(self.calibration_samples) < 10:

            print(
                "[WARNING] Not enough head-pose "
                "calibration samples"
            )

            return False

        samples = np.array(
            self.calibration_samples,
            dtype=np.float64
        )

        self.neutral_pose = np.median(
            samples,
            axis=0
        )

        self.calibration_samples.clear()

        # Reset smoothing after calibration.
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None

        # Reset temporal tracking.
        self.distraction_start = None
        self.distraction_duration = 0.0

        print(
            "[INFO] Head neutral position calibrated:",
            self.neutral_pose
        )

        return True

    # ======================================
    # CAMERA MATRIX
    # ======================================

    def create_camera_matrix(self, frame_shape):

        height, width = frame_shape[:2]

        focal_length = float(width)

        self.camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

    # ======================================
    # HEAD POSE ESTIMATION
    # ======================================

    def estimate_pose(self, landmarks, frame_shape):

        height, width = frame_shape[:2]

        if self.camera_matrix is None:

            self.create_camera_matrix(frame_shape)

        image_points = landmarks[
            self.LANDMARK_IDS, :2
        ].astype(np.float64)

        # Convert normalized coordinates to pixels.
        image_points[:, 0] *= width
        image_points[:, 1] *= height

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch, yaw, roll = [
            float(angle) for angle in angles
        ]

        # Handle equivalent Euler representation.
        pitch = self.normalize_angle(pitch)
        yaw = self.normalize_angle(yaw)
        roll = self.normalize_angle(roll)

        if pitch > 90:
            pitch -= 180

        elif pitch < -90:
            pitch += 180

        return yaw, pitch, roll

    # ======================================
    # DIRECTION CLASSIFICATION
    # ======================================

    def classify_direction(self, yaw, pitch):

        if abs(yaw) >= self.yaw_threshold:

            return (
                "RIGHT" if yaw > 0 else "LEFT"
            )

        if abs(pitch) >= self.pitch_threshold:

            return (
                "DOWN" if pitch > 0 else "UP"
            )

        return "FORWARD"

    # ======================================
    # TEMPORAL DISTRACTION
    # ======================================

    def update_distraction(self, direction, now):

        is_distracted = (
            direction != "FORWARD"
        )

        if is_distracted:

            if self.distraction_start is None:

                self.distraction_start = now

            self.distraction_duration = max(
                0.0,
                now - self.distraction_start
            )

        else:

            self.distraction_start = None
            self.distraction_duration = 0.0

        return self.distraction_duration

    # ======================================
    # MAIN PROCESS FUNCTION
    # ======================================

    def process(self, landmarks, frame_shape):

        now = time.monotonic()

        # Estimate raw pose.
        pose = self.estimate_pose(
            landmarks,
            frame_shape
        )

        if pose is None:
            return None

        yaw, pitch, roll = pose

        # Apply neutral calibration.
        if self.neutral_pose is not None:

            neutral_yaw, neutral_pitch, neutral_roll = (
                self.neutral_pose
            )

            yaw = self.normalize_angle(
                yaw - neutral_yaw
            )

            pitch = self.normalize_angle(
                pitch - neutral_pitch
            )

            roll = self.normalize_angle(
                roll - neutral_roll
            )

        # Smooth angles.
        self.smoothed_yaw = self.smooth_angle(
            self.smoothed_yaw,
            yaw
        )

        self.smoothed_pitch = self.smooth_angle(
            self.smoothed_pitch,
            pitch
        )

        self.smoothed_roll = self.smooth_angle(
            self.smoothed_roll,
            roll
        )

        yaw = self.smoothed_yaw
        pitch = self.smoothed_pitch
        roll = self.smoothed_roll

        # Calculate individual risks.
        yaw_risk = self.clamp(
            (
                abs(yaw) - self.yaw_threshold
            ) / 30.0
        )

        pitch_risk = self.clamp(
            (
                abs(pitch) - self.pitch_threshold
            ) / 25.0
        )

        roll_risk = self.clamp(
            (
                abs(roll) - 20.0
            ) / 30.0
        )

        # Determine direction.
        direction = self.classify_direction(
            yaw,
            pitch
        )

        # Track distraction duration.
        duration = self.update_distraction(
            direction,
            now
        )

        # Duration-based risk.
        duration_risk = self.clamp(
            duration / self.distraction_delay
        )

        # Combine angle and duration.
        angle_risk = max(
            yaw_risk,
            pitch_risk
        )

        distraction_score = (
            angle_risk *
            duration_risk
        )

        # ----------------------------------
        # FINAL OUTPUT
        # ----------------------------------

        return {

            "yaw": round(yaw, 2),

            "pitch": round(pitch, 2),

            "roll": round(roll, 2),

            "head_direction": direction,

            "yaw_risk": round(yaw_risk, 3),

            "pitch_risk": round(pitch_risk, 3),

            "roll_risk": round(roll_risk, 3),

            "distraction_duration": round(
                duration, 2
            ),

            "distraction_score": round(
                distraction_score, 3
            ),

            "is_distracted": (
                duration >= self.distraction_delay
            )
        }
