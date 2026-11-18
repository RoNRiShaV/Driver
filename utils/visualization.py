
import cv2


class Visualizer:

    def __init__(self, detector):
        self.detector = detector

    def draw_face_mesh(self, frame, face_data):

        if face_data is None:
            return frame

        face_landmarks = face_data["face_landmarks"]

        # Draw facial tessellation.
        self.detector.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.detector.mesh_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=
                self.detector.mp_drawing.DrawingSpec(
                    color=(80, 220, 120),
                    thickness=1,
                    circle_radius=1
                )
        )

        # Draw facial contours.
        self.detector.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.detector.contour_connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=
                self.detector.mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=2,
                    circle_radius=1
                )
        )

        return frame

    def draw_information(
        self,
        frame,
        fps,
        face_detected,
        landmark_count
    ):

        cv2.rectangle(
            frame,
            (10, 10),
            (410, 165),
            (25, 25, 25),
            -1
        )

        status = (
            "FACE DETECTED"
            if face_detected
            else "NO FACE DETECTED"
        )

        status_color = (
            (0, 255, 0)
            if face_detected
            else (0, 0, 255)
        )

        information = [
            ("DRIVER MONITORING SYSTEM", (255, 255, 255)),
            (f"FPS: {fps:.1f}", (255, 255, 255)),
            (f"STATUS: {status}", status_color),
            (f"LANDMARKS: {landmark_count}", (255, 255, 255)),
            ("Press Q to exit", (200, 200, 200))
        ]

        y_position = 35

        for text, color in information:

            cv2.putText(
                frame,
                text,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

            y_position += 28

        return frame
