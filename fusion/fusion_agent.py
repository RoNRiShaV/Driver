
import time
import numpy as np


class FusionAgent:

    def __init__(
        self,
        alpha=0.2,
        warning_threshold=0.40,
        drowsy_threshold=0.65,
        distracted_threshold=0.70,
        transition_delay=1.0,
        recovery_delay=2.0
    ):

        # ----------------------------------
        # WEIGHTED FUSION
        # ----------------------------------

        self.eye_weight = 0.35
        self.blink_weight = 0.40
        self.head_weight = 0.25

        self.alpha = alpha

        # ----------------------------------
        # STATE THRESHOLDS
        # ----------------------------------

        self.warning_threshold = warning_threshold
        self.drowsy_threshold = drowsy_threshold
        self.distracted_threshold = distracted_threshold

        self.transition_delay = transition_delay
        self.recovery_delay = recovery_delay

        # ----------------------------------
        # RISK SMOOTHING
        # ----------------------------------

        self.smoothed_risk = None

        # ----------------------------------
        # TEMPORAL STATE MANAGEMENT
        # ----------------------------------

        self.current_state = "SAFE"

        self.pending_state = None
        self.pending_start = None

        self.state_start = time.monotonic()

        self.last_alert = None

    @staticmethod
    def clamp(value):

        return float(
            np.clip(value, 0.0, 1.0)
        )

    # ======================================
    # WEIGHTED FUSION
    # ======================================

    def calculate_risk(
        self,
        eye_data,
        blink_data,
        head_data
    ):

        eye_risk = self.clamp(
            eye_data["eye_closure_risk"]
        )

        blink_risk = self.clamp(
            blink_data["fatigue_score"]
        )

        head_risk = self.clamp(
            head_data["distraction_score"]
        )

        raw_risk = (
            self.eye_weight * eye_risk +
            self.blink_weight * blink_risk +
            self.head_weight * head_risk
        )

        # Exponential moving average.
        if self.smoothed_risk is None:

            self.smoothed_risk = raw_risk

        else:

            self.smoothed_risk = (
                self.alpha * raw_risk +
                (1.0 - self.alpha) * self.smoothed_risk
            )

        # Separate risk categories.
        drowsiness_risk = (
            0.467 * eye_risk +
            0.533 * blink_risk
        )

        distraction_risk = head_risk

        return {
            "eye_risk": eye_risk,
            "blink_risk": blink_risk,
            "head_risk": head_risk,
            "raw_risk": raw_risk,
            "overall_risk": self.smoothed_risk,
            "drowsiness_risk": drowsiness_risk,
            "distraction_risk": distraction_risk
        }

    # ======================================
    # CANDIDATE STATE CLASSIFICATION
    # ======================================

    def classify_state(
        self,
        eye_data,
        blink_data,
        head_data,
        risk_data
    ):

        eye_risk = risk_data["eye_risk"]

        blink_risk = risk_data["blink_risk"]

        overall_risk = risk_data["overall_risk"]

        drowsiness_risk = risk_data["drowsiness_risk"]

        distraction_risk = risk_data["distraction_risk"]

        # Critical prolonged eye closure.
        current_closure = blink_data.get(
            "current_closure_duration",
            0.0
        )

        if current_closure >= 2.0:

            return "CRITICAL"

        # Severe eye closure risk.
        if eye_risk >= 0.98:

            return "CRITICAL"

        # Drowsiness.
        if drowsiness_risk >= self.drowsy_threshold:

            return "DROWSY"

        # Sustained head distraction.
        if (
            head_data.get("is_distracted", False)
            and distraction_risk >= 0.0
        ):

            return "DISTRACTED"

        # General warning.
        if overall_risk >= self.warning_threshold:

            return "WARNING"

        return "SAFE"

    # ======================================
    # TEMPORAL STATE MANAGER
    # ======================================

    def update_state(self, candidate_state, now):

        # Same state: clear pending transition.
        if candidate_state == self.current_state:

            self.pending_state = None
            self.pending_start = None

            return self.current_state, False

        # Critical alerts bypass normal confirmation.
        if candidate_state == "CRITICAL":

            old_state = self.current_state

            self.current_state = "CRITICAL"

            self.pending_state = None
            self.pending_start = None

            self.state_start = now

            return self.current_state, old_state != self.current_state

        # Start or update pending transition.
        if candidate_state != self.pending_state:

            self.pending_state = candidate_state
            self.pending_start = now

        elapsed = now - self.pending_start

        # Recovery requires a longer confirmation period.
        if candidate_state == "SAFE":

            required_time = self.recovery_delay

        else:

            required_time = self.transition_delay

        if elapsed >= required_time:

            old_state = self.current_state

            self.current_state = candidate_state

            self.pending_state = None
            self.pending_start = None

            self.state_start = now

            return self.current_state, old_state != self.current_state

        return self.current_state, False

    # ======================================
    # ALERT MANAGEMENT
    # ======================================

    def get_alert(self, state):

        alerts = {

            "SAFE": {
                "alert_level": "NORMAL",
                "alert_message": "Driver condition normal"
            },

            "WARNING": {
                "alert_level": "WARNING",
                "alert_message": "Please pay attention"
            },

            "DROWSY": {
                "alert_level": "HIGH",
                "alert_message": "Drowsiness detected - take a break"
            },

            "DISTRACTED": {
                "alert_level": "HIGH",
                "alert_message": "Look at the road"
            },

            "CRITICAL": {
                "alert_level": "CRITICAL",
                "alert_message": "Critical eye closure detected"
            },

            "INSUFFICIENT DATA": {
                "alert_level": "UNKNOWN",
                "alert_message": "Driver face not detected"
            }
        }

        return alerts[state]

    # ======================================
    # MAIN PROCESS FUNCTION
    # ======================================

    def process(
        self,
        eye_data,
        blink_data,
        head_data,
        timestamp=None
    ):

        now = (
            timestamp
            if timestamp is not None
            else time.monotonic()
        )

        # ----------------------------------
        # FACE / DATA VALIDATION
        # ----------------------------------

        if (
            eye_data is None
            or blink_data is None
            or head_data is None
        ):

            self.smoothed_risk = None

            self.pending_state = None
            self.pending_start = None

            self.current_state = "INSUFFICIENT DATA"

            alert = self.get_alert(
                "INSUFFICIENT DATA"
            )

            return {

                "driver_state": "INSUFFICIENT DATA",

                "overall_risk": None,

                "drowsiness_risk": None,

                "distraction_risk": None,

                "alert_level": alert["alert_level"],

                "alert_message": alert["alert_message"],

                "state_changed": False
            }

        # ----------------------------------
        # CALCULATE FUSED RISK
        # ----------------------------------

        risk_data = self.calculate_risk(
            eye_data,
            blink_data,
            head_data
        )

        # ----------------------------------
        # CLASSIFY CANDIDATE STATE
        # ----------------------------------

        candidate_state = self.classify_state(
            eye_data,
            blink_data,
            head_data,
            risk_data
        )

        # ----------------------------------
        # TEMPORAL STATE UPDATE
        # ----------------------------------

        driver_state, state_changed = (
            self.update_state(
                candidate_state,
                now
            )
        )

        # ----------------------------------
        # STATE DURATION
        # ----------------------------------

        state_duration = max(
            0.0,
            now - self.state_start
        )

        # ----------------------------------
        # ALERT GENERATION
        # ----------------------------------

        alert = self.get_alert(
            driver_state
        )

        # Print only when state changes.
        if state_changed:

            print(
                f"[DRIVER STATE] {driver_state}"
            )

            print(
                f"[ALERT] {alert['alert_message']}"
            )

        self.last_alert = alert["alert_level"]

        # ----------------------------------
        # RETURN FUSION OUTPUT
        # ----------------------------------

        return {

            "eye_risk": round(
                risk_data["eye_risk"], 3
            ),

            "blink_risk": round(
                risk_data["blink_risk"], 3
            ),

            "head_risk": round(
                risk_data["head_risk"], 3
            ),

            "raw_risk": round(
                risk_data["raw_risk"], 3
            ),

            "overall_risk": round(
                risk_data["overall_risk"], 3
            ),

            "drowsiness_risk": round(
                risk_data["drowsiness_risk"], 3
            ),

            "distraction_risk": round(
                risk_data["distraction_risk"], 3
            ),

            "candidate_state": candidate_state,

            "driver_state": driver_state,

            "state_duration": round(
                state_duration, 2
            ),

            "state_changed": state_changed,

            "alert_level": alert["alert_level"],

            "alert_message": alert["alert_message"]
        }
