
from collections import deque
import time
import numpy as np


class BlinkAgent:

    def __init__(
        self,
        window_seconds=60,
        min_blink_duration=0.08,
        max_blink_duration=0.80,
        warmup_seconds=10.0
    ):

        self.window_seconds = window_seconds
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self.warmup_seconds = warmup_seconds

        # Timestamped eye-state observations.
        self.observations = deque()

        # Completed blink information.
        self.blink_timestamps = deque()
        self.blink_durations = deque(maxlen=100)

        # Prolonged closure information.
        self.long_closure_timestamps = deque()

        # Current eye closure state.
        self.was_closed = False
        self.closure_start = None

        self.total_blinks = 0
        self.total_long_closures = 0

        self.session_start = None

    @staticmethod
    def clamp(value):

        return float(
            np.clip(value, 0.0, 1.0)
        )

    def _update_observations(self, now, eye_closed):

        self.observations.append(
            (now, eye_closed)
        )

        cutoff = now - self.window_seconds

        # Retain one observation before the cutoff
        # to correctly calculate the overlapping interval.
        while (
            len(self.observations) >= 2
            and self.observations[1][0] <= cutoff
        ):

            self.observations.popleft()

    def _calculate_perclos(self, now):

        if len(self.observations) < 2:
            return 0.0, 0.0

        cutoff = now - self.window_seconds

        closed_time = 0.0
        observed_time = 0.0

        samples = list(self.observations)

        for i in range(len(samples) - 1):

            previous_time, previous_closed = samples[i]
            current_time, _ = samples[i + 1]

            interval_start = max(previous_time, cutoff)
            interval_end = min(current_time, now)

            duration = max(
                0.0,
                interval_end - interval_start
            )

            observed_time += duration

            if previous_closed:
                closed_time += duration

        # Include the time since the latest observation.
        # This accounts for an ongoing eye closure.
        last_time, last_closed = samples[-1]

        interval_start = max(last_time, cutoff)

        duration = max(
            0.0,
            now - interval_start
        )

        observed_time += duration

        if last_closed:
            closed_time += duration

        if observed_time <= 0:
            return 0.0, 0.0

        perclos = closed_time / observed_time

        return (
            self.clamp(perclos),
            observed_time
        )

    def _cleanup_events(self, now):

        cutoff = now - self.window_seconds

        while (
            self.blink_timestamps
            and self.blink_timestamps[0] < cutoff
        ):

            self.blink_timestamps.popleft()

        while (
            self.long_closure_timestamps
            and self.long_closure_timestamps[0] < cutoff
        ):

            self.long_closure_timestamps.popleft()

    def process(self, eye_data, timestamp=None):

        now = (
            timestamp
            if timestamp is not None
            else time.monotonic()
        )

        if self.session_start is None:
            self.session_start = now

        # Determine current eye state.
        eye_state = eye_data["eye_state"]

        eye_closed = (
            eye_state == "CLOSED"
        )

        # --------------------------------------
        # 1. STORE OBSERVATIONS
        # --------------------------------------

        self._update_observations(
            now,
            eye_closed
        )

        # --------------------------------------
        # 2. DETECT BLINK EVENTS
        # --------------------------------------

        if eye_closed and not self.was_closed:

            # Start of eye closure.
            self.closure_start = now

        elif not eye_closed and self.was_closed:

            # End of eye closure.
            if self.closure_start is not None:

                duration = max(
                    0.0,
                    now - self.closure_start
                )

                if (
                    self.min_blink_duration
                    <= duration
                    <= self.max_blink_duration
                ):

                    # Valid completed blink.
                    self.total_blinks += 1

                    self.blink_timestamps.append(now)

                    self.blink_durations.append(
                        duration
                    )

                elif duration > self.max_blink_duration:

                    # Prolonged eye closure.
                    self.total_long_closures += 1

                    self.long_closure_timestamps.append(
                        now
                    )

            self.closure_start = None

        self.was_closed = eye_closed

        # --------------------------------------
        # 3. CALCULATE CURRENT CLOSURE DURATION
        # --------------------------------------

        current_closure_duration = 0.0

        if self.was_closed and self.closure_start is not None:

            current_closure_duration = max(
                0.0,
                now - self.closure_start
            )

        # --------------------------------------
        # 4. CLEANUP ROLLING WINDOW
        # --------------------------------------

        self._cleanup_events(now)

        # --------------------------------------
        # 5. CALCULATE PERCLOS
        # --------------------------------------

        perclos, observed_time = (
            self._calculate_perclos(now)
        )

        # --------------------------------------
        # 6. BLINK RATE
        # --------------------------------------

        effective_window = min(
            self.window_seconds,
            observed_time
        )

        if effective_window > 0:

            blink_rate = (
                len(self.blink_timestamps)
                / effective_window
            ) * 60.0

        else:

            blink_rate = 0.0

        # --------------------------------------
        # 7. AVERAGE BLINK DURATION
        # --------------------------------------

        if self.blink_durations:

            average_duration = float(
                np.mean(self.blink_durations)
            )

        else:

            average_duration = 0.0

        # --------------------------------------
        # 8. FATIGUE RISK COMPONENTS
        # --------------------------------------

        perclos_risk = self.clamp(
            perclos / 0.40
        )

        duration_risk = self.clamp(
            average_duration / 0.60
        )

        # Risk from completed prolonged closures.
        long_closure_risk = self.clamp(
            len(self.long_closure_timestamps) / 3.0
        )

        # Risk from a currently ongoing closure.
        ongoing_closure_risk = self.clamp(
            current_closure_duration / 1.5
        )

        # Use the more severe closure indicator.
        closure_risk = max(
            long_closure_risk,
            ongoing_closure_risk
        )

        # --------------------------------------
        # 9. ABNORMAL BLINK RATE
        # --------------------------------------

        elapsed_session = now - self.session_start

        if elapsed_session >= self.warmup_seconds:

            if blink_rate < 10:

                blink_rate_risk = self.clamp(
                    (10 - blink_rate) / 10
                )

            elif blink_rate > 30:

                blink_rate_risk = self.clamp(
                    (blink_rate - 30) / 30
                )

            else:

                blink_rate_risk = 0.0

        else:

            # Do not infer abnormal blink rate
            # before enough observations exist.
            blink_rate_risk = 0.0

        # --------------------------------------
        # 10. FINAL FATIGUE SCORE
        # --------------------------------------

        fatigue_score = (
            0.40 * perclos_risk +
            0.20 * duration_risk +
            0.30 * closure_risk +
            0.10 * blink_rate_risk
        )

        fatigue_score = self.clamp(
            fatigue_score
        )

        # --------------------------------------
        # 11. RETURN AGENT OUTPUT
        # --------------------------------------

        return {

            "blink_count": self.total_blinks,

            "blink_rate": round(
                blink_rate, 2
            ),

            "average_blink_duration": round(
                average_duration, 3
            ),

            "perclos": round(
                perclos, 4
            ),

            "long_closures": self.total_long_closures,

            "current_closure_duration": round(
                current_closure_duration, 3
            ),

            "fatigue_score": round(
                fatigue_score, 4
            ),

            "alertness_score": round(
                1.0 - fatigue_score, 4
            ),

            "data_ready": (
                elapsed_session >= self.warmup_seconds
            )
        }
