import cv2

from utils.hand_processor import HandProcessor
from utils.webcam import Webcam
from utils.classifier import Classifier

WIN_NAME = 'ASL Recognition'


def main():
    phrase = ''
    hand_processor = HandProcessor()
    webcam = Webcam()
    classifier = Classifier('model/asl_classifier_model.keras')

    while True:
        image = webcam.read_frame()
        if image is None:
            break

        results = hand_processor.process_frame(image)
        # image = hand_processor.draw_landmarks(image, results)

        prediction = None
        try: 
            prediction = classifier.predict(results)
        except:
            pass

        if phrase != '':
                cv2.putText(
                    image,
                    f'Phrase: {phrase}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        if prediction is not None:
            cv2.putText(
                image,
                f'Prediction: {chr(prediction + 65)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # on space key press, save the character
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                char = chr(prediction + 65)

                if char == '[':
                    phrase = phrase[:-1]
                else:
                    if char == '\\':
                        char = ' '
                    
                    phrase += char
                    

        cv2.imshow(WIN_NAME, image)

        # If the 'q' key is pressed or the window closed.
        if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    webcam.release()


if __name__ == '__main__':
    main()
