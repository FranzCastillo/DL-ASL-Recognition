import cv2

from utils.hand_processor import HandProcessor
from utils.webcam import Webcam

WIN_NAME = 'ASL Recognition'


def main():
    hand_processor = HandProcessor()
    webcam = Webcam()

    while True:
        image = webcam.read_frame()
        if image is None:
            break

        results = hand_processor.process_frame(image)
        image = hand_processor.draw_landmarks(image, results)

        cv2.imshow(WIN_NAME, image)

        # If the 'q' key is pressed or the window closed.
        if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    webcam.release()


if __name__ == '__main__':
    main()
