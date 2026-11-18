
import csv
import os
import time
from datetime import datetime


class EvaluationLogger:
    """
    Records agent outputs, fusion predictions and manually assigned
    ground-truth labels for Driver Monitoring System evaluation.
    """

    def __init__(self, output_dir="evaluation_results", sample_interval=1.0):
        os.makedirs(output_dir, exist_ok=True)

        filename = "evaluation_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.csv_path = os.path.join(output_dir, filename)

        self.start_time = time.time()
        self.sample_interval = float(sample_interval)
        self.last_sample_time = 0.0

        self.ground_truth = "UNLABELLED"

        self.observation_count = 0
        self.state_transitions = 0
        self.previous_state = None

        self.total_drowsy_events = 0
        self.detected_drowsy_events = 0
        self.missed_drowsy_events = 0

        self.reaction_times = []
        self.active_event = None

        self.fieldnames = [
            "timestamp",
            "elapsed_time",
            "ground_truth",
            "eye_state",
            "left_ear",
            "right_ear",
            "eye_closure_risk",
            "blink_count",
            "blink_rate",
            "average_blink_duration",
            "perclos",
            "fatigue_score",
            "long_closures",
            "current_closure_duration",
            "yaw",
            "pitch",
            "roll",
            "head_direction",
            "distraction_duration",
            "distraction_score",
            "is_distracted",
            "raw_risk",
            "overall_risk",
            "drowsiness_risk",
            "distraction_risk",
            "candidate_state",
            "driver_state",
            "alert_level",
            "state_changed",
            "event_detected",
            "reaction_time"
        ]

        self.file = open(
            self.csv_path,
            mode="w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.fieldnames
        )
        self.writer.writeheader()
        self.file.flush()

        print("\n[Evaluation] Logging to:", self.csv_path)
        print("[Evaluation] Press 1=NORMAL, 2=DROWSY, 3=DISTRACTED,")
        print("[Evaluation]       4=CRITICAL, 0=UNLABELLED\n")

    @staticmethod
    def get(data, *keys, default=""):
        """Return the first available key from a dictionary."""
        if not isinstance(data, dict):
            return default

        for key in keys:
            value = data.get(key)
            if value is not None:
                return value

        return default

    def set_ground_truth(self, label):
        """
        Set the manually observed condition.

        A new drowsiness event begins when the label changes from a
        non-drowsy label to DROWSY or CRITICAL.
        """
        label = str(label).upper()

        valid_labels = {
            "NORMAL",
            "DROWSY",
            "DISTRACTED",
            "CRITICAL",
            "UNLABELLED"
        }

        if label not in valid_labels:
            print("[Evaluation] Invalid label:", label)
            return

        if label == self.ground_truth:
            return

        previous_label = self.ground_truth
        self.ground_truth = label

        is_drowsy_label = label in ("DROWSY", "CRITICAL")
        was_drowsy_label = previous_label in ("DROWSY", "CRITICAL")

        if is_drowsy_label and not was_drowsy_label:
            self.total_drowsy_events += 1

            self.active_event = {
                "start_time": time.time(),
                "detected": False,
                "pre_existing_detection": False
            }

            print("[Evaluation] Drowsiness event started:", label)

        elif not is_drowsy_label and was_drowsy_label:
            self._finish_active_event()

        print("[Evaluation] Ground truth:", label)

    def _finish_active_event(self):
        """Close the current manually annotated drowsiness event."""
        if self.active_event is None:
            return

        if not self.active_event["detected"]:
            self.missed_drowsy_events += 1
            print("[Evaluation] Event ended without detection.")

        self.active_event = None

    def _check_event_detection(self, driver_state):
        """
        Detect the first DROWSY or CRITICAL prediction during an
        annotated drowsiness event.
        """
        if self.active_event is None:
            return False, ""

        if self.active_event["detected"]:
            return False, ""

        if driver_state not in ("DROWSY", "CRITICAL"):
            return False, ""

        elapsed = time.time() - self.active_event["start_time"]

        self.active_event["detected"] = True
        self.detected_drowsy_events += 1

        # If the system was already predicting drowsiness when the
        # manual annotation began, this is not a valid reaction-time
        # measurement.
        if self.previous_state in ("DROWSY", "CRITICAL"):
            self.active_event["pre_existing_detection"] = True
            print(
                "[Evaluation] Event was already detected at annotation "
                "time; reaction time excluded."
            )
            return True, ""

        self.reaction_times.append(elapsed)

        print(
            "[Evaluation] Drowsiness detected. Reaction time: "
            "{:.3f} seconds".format(elapsed)
        )

        return True, elapsed

    def log(self, eye_data, blink_data, head_data, fusion_data):
        """Process one frame's outputs and write a sampled CSV row."""

        now = time.time()
        elapsed = now - self.start_time

        driver_state = self.get(
            fusion_data,
            "driver_state",
            default="INSUFFICIENT DATA"
        )

        if not driver_state:
            driver_state = "INSUFFICIENT DATA"

        driver_state = str(driver_state).upper()

        # Count state transitions at frame level.
        if self.previous_state is not None:
            if driver_state != self.previous_state:
                self.state_transitions += 1

        self.observation_count += 1

        event_detected, reaction_time = self._check_event_detection(
            driver_state
        )

        self.previous_state = driver_state

        # Write one representative observation per sample interval.
        if now - self.last_sample_time < self.sample_interval:
            return

        self.last_sample_time = now

        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_time": round(elapsed, 3),
            "ground_truth": self.ground_truth,

            "eye_state": self.get(
                eye_data, "eye_state", "state"
            ),
            "left_ear": self.get(
                eye_data, "left_ear", "left_EAR", "left_eye_ear"
            ),
            "right_ear": self.get(
                eye_data, "right_ear", "right_EAR", "right_eye_ear"
            ),
            "eye_closure_risk": self.get(
                eye_data, "eye_closure_risk", "eye_risk"
            ),

            "blink_count": self.get(
                blink_data, "blink_count", "total_blinks"
            ),
            "blink_rate": self.get(
                blink_data, "blink_rate", "blinks_per_minute"
            ),
            "average_blink_duration": self.get(
                blink_data,
                "average_blink_duration",
                "avg_blink_duration"
            ),
            "perclos": self.get(
                blink_data, "perclos", "perclos_value"
            ),
            "fatigue_score": self.get(
                blink_data, "fatigue_score"
            ),
            "long_closures": self.get(
                blink_data, "long_closures", "long_closure_count"
            ),
            "current_closure_duration": self.get(
                blink_data, "current_closure_duration"
            ),

            "yaw": self.get(head_data, "yaw"),
            "pitch": self.get(head_data, "pitch"),
            "roll": self.get(head_data, "roll"),
            "head_direction": self.get(
                head_data, "head_direction", "direction"
            ),
            "distraction_duration": self.get(
                head_data, "distraction_duration"
            ),
            "distraction_score": self.get(
                head_data, "distraction_score"
            ),
            "is_distracted": self.get(
                head_data, "is_distracted"
            ),

            "raw_risk": self.get(fusion_data, "raw_risk"),
            "overall_risk": self.get(fusion_data, "overall_risk"),
            "drowsiness_risk": self.get(
                fusion_data, "drowsiness_risk"
            ),
            "distraction_risk": self.get(
                fusion_data, "distraction_risk"
            ),
            "candidate_state": self.get(
                fusion_data, "candidate_state"
            ),
            "driver_state": driver_state,
            "alert_level": self.get(fusion_data, "alert_level"),
            "state_changed": self.get(
                fusion_data, "state_changed"
            ),
            "event_detected": event_detected,
            "reaction_time": reaction_time
        }

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the CSV and print a basic evaluation summary."""
        if self.file.closed:
            return

        # An event still active at shutdown is counted as missed if
        # the system never detected it.
        self._finish_active_event()

        self.file.flush()
        self.file.close()

        stability = None
        if self.observation_count > 1:
            stability = 1.0 - (
                self.state_transitions / (self.observation_count - 1)
            )

        print("\n========== EVALUATION SUMMARY ==========")
        print("CSV file:", self.csv_path)
        print("Frame observations:", self.observation_count)
        print("State transitions:", self.state_transitions)

        if stability is not None:
            print("Frame-level stability: {:.4f}".format(stability))

        print("Total annotated drowsiness events:",
              self.total_drowsy_events)
        print("Detected drowsiness events:",
              self.detected_drowsy_events)
        print("Missed drowsiness events:",
              self.missed_drowsy_events)

        if self.reaction_times:
            print(
                "Average reaction time: {:.3f} seconds".format(
                    sum(self.reaction_times) / len(self.reaction_times)
                )
            )
            print(
                "Minimum reaction time: {:.3f} seconds".format(
                    min(self.reaction_times)
                )
            )
            print(
                "Maximum reaction time: {:.3f} seconds".format(
                    max(self.reaction_times)
                )
            )
        else:
            print("No valid reaction-time measurements recorded.")

        print("========================================\n")
