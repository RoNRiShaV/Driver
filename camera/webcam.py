
import cv2


class Webcam:

    def __init__(
        self,
        camera_index=0,
        width=480,
        height=360,
        fps=30
    ):

        self.cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        # Fallback camera backend.
        if not self.cap.isOpened():

            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open webcam. "
                "Check camera permissions or camera index."
            )

        # Camera settings.
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            fps
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        print("[INFO] Webcam initialized")
        print(
            f"[INFO] Resolution: {self.width}x{self.height}"
        )

        print(
            f"[INFO] Requested FPS: {fps}"
        )

    def read(self):

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):

        if self.cap is not None:

            self.cap.release()

            print("[INFO] Webcam released")
