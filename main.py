
import cv2
import time

from utils.evaluation_logger import EvaluationLogger

from camera.webcam import Webcam
from camera.face_mesh import FaceMeshDetector

from agents.eye_state_agent import EyeStateAgent
from agents.blink_agent import BlinkAgent
from agents.head_pose_agent import HeadPoseAgent

from fusion.fusion_agent import FusionAgent


DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600


# ==========================================
# TEXT HELPER
# ==========================================

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.42
):
    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )


# ==========================================
# COMPACT DASHBOARD
# ==========================================

def draw_dashboard(
    frame,
    fps,
    face_data,
    eye_data,
    blink_data,
    head_data,
    fusion_data,
    calibration_message,
    ground_truth="UNLABELLED"
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (290, 455),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame
    )

    # ======================================
    # SYSTEM INFORMATION
    # ======================================

    draw_text(
        frame,
        "DRIVER MONITORING SYSTEM",
        (15, 30),
        (0, 255, 255),
        0.43
    )

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (15, 57)
    )

    face_detected = face_data is not None

    draw_text(
        frame,
        f"FACE: {'DETECTED' if face_detected else 'NOT DETECTED'}",
        (15, 84),
        (0, 255, 0) if face_detected else (0, 0, 255)
    )

    # ======================================
    # EYE STATE AGENT
    # ======================================

    draw_text(
        frame,
        "--- EYE STATE AGENT ---",
        (15, 120),
        (0, 255, 255)
    )

    if eye_data is not None:

        draw_text(
            frame,
            f"Left EAR: {eye_data['left_ear']:.3f}",
            (15, 148)
        )

        draw_text(
            frame,
            f"Right EAR: {eye_data['right_ear']:.3f}",
            (15, 175)
        )

        draw_text(
            frame,
            f"Eye State: {eye_data['eye_state']}",
            (15, 202)
        )

        draw_text(
            frame,
            f"Eye Risk: {eye_data['eye_closure_risk']:.2f}",
            (15, 229)
        )

    # ======================================
    # BLINK AGENT
    # ======================================

    draw_text(
        frame,
        "--- BLINK AGENT ---",
        (15, 270),
        (0, 255, 255)
    )

    if blink_data is not None:

        draw_text(
            frame,
            f"Blinks: {blink_data['blink_count']}",
            (15, 298)
        )

        draw_text(
            frame,
            f"Rate: {blink_data['blink_rate']:.1f}/min",
            (15, 325)
        )

        draw_text(
            frame,
            f"PERCLOS: {blink_data['perclos']:.3f}",
            (15, 352)
        )

        draw_text(
            frame,
            f"Fatigue: {blink_data['fatigue_score']:.2f}",
            (15, 379)
        )

    # ======================================
    # HEAD POSE AGENT
    # ======================================

    draw_text(
        frame,
        "--- HEAD POSE AGENT ---",
        (15, 414),
        (0, 255, 255)
    )

    if head_data is not None:

        draw_text(
            frame,
            f"Direction: {head_data['head_direction']}",
            (15, 437)
        )

    # ======================================
    # FUSION AGENT
    # ======================================

    if fusion_data is not None:

        state = fusion_data["driver_state"]
        risk = fusion_data["overall_risk"]

        if state == "SAFE":
            state_color = (0, 255, 0)

        elif state == "INSUFFICIENT DATA":
            state_color = (0, 165, 255)

        else:
            state_color = (0, 0, 255)

        draw_text(
            frame,
            f"FUSION: {state}",
            (15, 475),
            state_color
        )

        if risk is not None:
            draw_text(
                frame,
                f"Overall Risk: {risk:.2f}",
                (310, 110),
                (255, 255, 255)
            )

    # ======================================
    # HEAD POSE DETAILS
    # ======================================

    if head_data is not None:

        draw_text(
            frame,
            f"Yaw: {head_data['yaw']:.1f}",
            (310, 30)
        )

        draw_text(
            frame,
            f"Pitch: {head_data['pitch']:.1f}",
            (310, 55)
        )

        draw_text(
            frame,
            f"Distraction: {head_data['distraction_score']:.2f}",
            (310, 80)
        )

    # ======================================
    # CALIBRATION STATUS
    # ======================================

    draw_text(
        frame,
        calibration_message,
        (310, 450),
        (0, 255, 255)
    )

    # ======================================
    # EVALUATION STATUS
    # ======================================

    draw_text(
        frame,
        f"GROUND TRUTH: {ground_truth}",
        (310, 475),
        (0, 255, 0)
    )

    draw_text(
        frame,
        "1: Normal | 2: Drowsy | 3: Distracted",
        (310, 500),
        (255, 255, 255),
        0.38
    )

    draw_text(
        frame,
        "4: Critical | 0: Unlabelled | C: Calibrate | Q: Exit",
        (310, 525),
        (255, 255, 255),
        0.36
    )

    return frame


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    webcam = None
    detector = None
    evaluation_logger = None

    try:

        # ==================================
        # WEBCAM INITIALIZATION
        # ==================================

        webcam = Webcam(
            camera_index=0,
            width=640,
            height=480,
            fps=30
        )

        # ==================================
        # MEDIAPIPE INITIALIZATION
        # ==================================

        detector = FaceMeshDetector(
            max_faces=1,
            detection_confidence=0.6,
            tracking_confidence=0.6
        )

        # ==================================
        # AGENT INITIALIZATION
        # ==================================

        eye_agent = EyeStateAgent()

        blink_agent = BlinkAgent(
            window_seconds=60
        )

        head_agent = HeadPoseAgent()

        # ==================================
        # FUSION AGENT
        # ==================================

        fusion_agent = FusionAgent(
            alpha=0.2
        )

        # ==================================
        # EVALUATION LOGGER
        # ==================================

        evaluation_logger = EvaluationLogger(
            output_dir="evaluation_results",
            sample_interval=1.0
        )

        # ==================================
        # DISPLAY WINDOW
        # ==================================

        cv2.namedWindow(
            "Driver Monitoring System",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Driver Monitoring System",
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        # ==================================
        # FPS TRACKING
        # ==================================

        previous_time = time.perf_counter()
        fps = 0.0

        # ==================================
        # CALIBRATION SETTINGS
        # ==================================

        calibration_active = False
        calibration_start = 0.0
        calibration_duration = 3.0

        calibration_message = "Press C to calibrate"

        print("\n[INFO] Driver Monitoring System started")
        print("[INFO] Press C to calibrate")
        print("[INFO] Evaluation keyboard controls:")
        print("[INFO] 1 = NORMAL")
        print("[INFO] 2 = DROWSY")
        print("[INFO] 3 = DISTRACTED")
        print("[INFO] 4 = CRITICAL")
        print("[INFO] 0 = UNLABELLED")
        print("[INFO] Q = EXIT\n")

        # ==================================
        # MAIN LOOP
        # ==================================

        while True:

            frame = webcam.read()

            if frame is None:
                print("[WARNING] Webcam frame unavailable")
                break

            # Mirror webcam.
            frame = cv2.flip(frame, 1)

            # ==================================
            # FACE DETECTION
            # ==================================

            face_data = detector.process(frame)

            now = time.monotonic()

            eye_data = None
            blink_data = None
            head_data = None

            # ==================================
            # FACE STATUS
            # ==================================

            if not calibration_active:

                calibration_message = (
                    "FACE DETECTED"
                    if face_data is not None
                    else "FACE NOT DETECTED"
                )

            # ==================================
            # THREE AGENTS
            # ==================================

            if face_data is not None:

                landmarks = face_data["landmarks"]

                # ----------------------------------
                # AGENT 1: EYE STATE
                # ----------------------------------

                if calibration_active:

                    eye_agent.calibrate(landmarks)

                eye_data = eye_agent.process(
                    landmarks
                )

                # ----------------------------------
                # AGENT 2: BLINK PATTERN
                # ----------------------------------

                blink_data = blink_agent.process(
                    eye_data,
                    timestamp=now
                )

                # ----------------------------------
                # AGENT 3: HEAD POSE
                # ----------------------------------

                head_data = head_agent.process(
                    landmarks,
                    frame.shape
                )

                if calibration_active and head_data is not None:

                    head_agent.calibrate(
                        head_data["yaw"],
                        head_data["pitch"],
                        head_data["roll"]
                    )

                # ----------------------------------
                # CALIBRATION
                # ----------------------------------

                if calibration_active:

                    elapsed = now - calibration_start

                    remaining = max(
                        0,
                        calibration_duration - elapsed
                    )

                    calibration_message = (
                        f"CALIBRATING: {remaining:.1f}s"
                    )

                    if elapsed >= calibration_duration:

                        eye_success = (
                            eye_agent.finish_calibration()
                        )

                        head_success = (
                            head_agent.finish_calibration()
                        )

                        calibration_active = False

                        if eye_success and head_success:

                            calibration_message = (
                                "CALIBRATION COMPLETE"
                            )

                            print("[INFO] Calibration successful")

                        else:

                            calibration_message = (
                                "CALIBRATION FAILED - RETRY"
                            )

                            print("[WARNING] Calibration failed")

                # ----------------------------------
                # DRAW FACIAL LANDMARKS
                # ----------------------------------

                detector.mp_drawing.draw_landmarks(
                    frame,
                    face_data["face_landmarks"],
                    detector.contour_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                        detector.mp_drawing.DrawingSpec(
                            color=(0, 255, 255),
                            thickness=1
                        )
                )

            # ==================================
            # FUSION AGENT
            # ==================================

            fusion_data = fusion_agent.process(
                eye_data,
                blink_data,
                head_data
            )

            # ==================================
            # EVALUATION LOGGING
            # ==================================

            if evaluation_logger is not None:

                evaluation_logger.log(
                    eye_data,
                    blink_data,
                    head_data,
                    fusion_data
                )

            # ==================================
            # FPS CALCULATION
            # ==================================

            current_time = time.perf_counter()

            delta_time = current_time - previous_time

            if delta_time > 0:

                instant_fps = 1.0 / delta_time

                fps = (
                    0.9 * fps + 0.1 * instant_fps
                    if fps > 0
                    else instant_fps
                )

            previous_time = current_time

            # ==================================
            # DRAW DASHBOARD
            # ==================================

            frame = draw_dashboard(
                frame,
                fps,
                face_data,
                eye_data,
                blink_data,
                head_data,
                fusion_data,
                calibration_message,
                evaluation_logger.ground_truth
            )

            # ==================================
            # DISPLAY FRAME
            # ==================================

            display_frame = cv2.resize(
                frame,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_LINEAR
            )

            cv2.imshow(
                "Driver Monitoring System",
                display_frame
            )

            # ==================================
            # KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                break

            elif key == ord('c') and not calibration_active:

                eye_agent.calibration_samples.clear()

                head_agent.calibration_samples.clear()

                calibration_active = True

                calibration_start = time.monotonic()

                calibration_message = "CALIBRATION STARTED"

                print("[INFO] Calibration started")
                print("[INFO] Keep eyes open and face forward")

            # ==================================
            # GROUND-TRUTH LABELS
            # ==================================

            elif key == ord('1'):

                evaluation_logger.set_ground_truth(
                    "NORMAL"
                )

            elif key == ord('2'):

                evaluation_logger.set_ground_truth(
                    "DROWSY"
                )

            elif key == ord('3'):

                evaluation_logger.set_ground_truth(
                    "DISTRACTED"
                )

            elif key == ord('4'):

                evaluation_logger.set_ground_truth(
                    "CRITICAL"
                )

            elif key == ord('0'):

                evaluation_logger.set_ground_truth(
                    "UNLABELLED"
                )

    except KeyboardInterrupt:

        print("[INFO] Interrupted by user")

    except Exception as error:

        print("[ERROR]", error)
        raise

    finally:

        # ==================================
        # CLOSE EVALUATION LOGGER
        # ==================================

        if evaluation_logger is not None:

            evaluation_logger.close()

        # ==================================
        # RELEASE CAMERA
        # ==================================

        if detector is not None:
            detector.close()

        if webcam is not None:
            webcam.release()

        # ==================================
        # DESTROY WINDOWS
        # ==================================

        cv2.destroyAllWindows()

        print("[INFO] System stopped")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
