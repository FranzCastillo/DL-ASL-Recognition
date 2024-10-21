import os

import cv2

from utils.webcam import Webcam

DATA_DIR = 'data'
PICTURES_PER_CLASS = 50
WIN_NAME = 'Collecting ASL Data'


def _create_folders():
    """
    Create folders for each letter of the alphabet, space and delete.
    :param clear_existing: Boolean indicating whether to clear existing folders.
    :return:
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for letter in range(65, 91):  # A-Z
        letter = chr(letter)
        if not os.path.exists(f'{DATA_DIR}/{letter}'):
            os.makedirs(f'{DATA_DIR}/{letter}')

    if not os.path.exists(f'{DATA_DIR}/space'):
        os.makedirs(f'{DATA_DIR}/space')

    if not os.path.exists(f'{DATA_DIR}/delete'):
        os.makedirs(f'{DATA_DIR}/delete')


def main():
    cam = Webcam()

    _create_folders()

    classes = os.listdir(DATA_DIR)  # Amount of folders in DATA_DIR
    print(f'Number of classes: {len(classes)}')

    for i, class_name in enumerate(classes):
        while True:  # When the user is ready, press SPACE to start
            image = cam.read_frame()
            if image is None:
                break

            # Add text
            cv2.putText(
                image,
                f'Press SPACE to start capturing {class_name}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow(WIN_NAME, image)

            key = cv2.waitKey(25) & 0xFF
            if key == ord(' '):  # If the 'SPACE' key is pressed.
                break
            elif key == ord('q'):  # If the 'Q' key is pressed.
                cam.release()
                return

        # Start taking pictures.
        for j in range(PICTURES_PER_CLASS):
            image = cam.read_frame()
            if image is None:
                break

            # Display the image.
            cv2.imshow(WIN_NAME, image)

            # Save the image to the corresponding class folder.
            cv2.imwrite(f'{DATA_DIR}/{class_name}/{j}.jpg', image)

            # Wait 25ms for the next picture to be taken.
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):  # If the 'Q' key is pressed.
                cam.release()
                return

    cam.release()


if __name__ == '__main__':
    main()
