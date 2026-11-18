
import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:

    def __init__(
        self,
        max_faces=1,
        detection_confidence=0.6,
        tracking_confidence=0.6
    ):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils

        self.mesh_connections = (
            self.mp_face_mesh.FACEMESH_TESSELATION
        )

        self.contour_connections = (
            self.mp_face_mesh.FACEMESH_CONTOURS
        )

        print("[INFO] Face Mesh initialized")

    def process(self, frame):

        # OpenCV uses BGR, MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False

        results = self.face_mesh.process(rgb_frame)

        rgb_frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        # We only monitor the first detected face.
        face_landmarks = results.multi_face_landmarks[0]

        height, width, _ = frame.shape

        landmarks = []

        for landmark in face_landmarks.landmark:

            x = landmark.x
            y = landmark.y
            z = landmark.z

            landmarks.append([x, y, z])

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )

        # Convert normalized coordinates into pixels.
        landmarks_px = np.zeros(
            (len(landmarks), 2),
            dtype=np.int32
        )

        landmarks_px[:, 0] = (
            landmarks[:, 0] * width
        ).astype(np.int32)

        landmarks_px[:, 1] = (
            landmarks[:, 1] * height
        ).astype(np.int32)

        return {
            "face_landmarks": face_landmarks,
            "landmarks": landmarks,
            "landmarks_px": landmarks_px,
            "landmark_count": len(landmarks)
        }

    def close(self):
        self.face_mesh.close()
        print("[INFO] Face Mesh released")
