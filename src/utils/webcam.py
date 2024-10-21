import cv2


class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def read_frame(self):
        success, image = self.cap.read()
        if not success:
            return None
        image = cv2.flip(image, 1)  # Mirror the image.
        return image

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
