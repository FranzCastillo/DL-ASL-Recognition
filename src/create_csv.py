import argparse
import csv
import os

from utils.hand_processor import HandProcessor

DATA_DIR = 'data'
CSV_PATH = 'model/data'


def main(filename):
    output_path = os.path.join(CSV_PATH, filename)

    landmarker = HandProcessor(
        static_image_mode=True,
        min_detection_confidence=0.4
    )

    data = []

    classes = os.listdir(DATA_DIR)
    for class_name in classes:
        print(f'Processing class: {class_name}')
        class_dir = f'{DATA_DIR}/{class_name}'
        images = os.listdir(class_dir)

        if len(images) == 0:
            print(f'No images found')
            continue

        for image in images:
            image_path = f'{class_dir}/{image}'
            results = landmarker.process_image(image_path)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y])
                    landmarks.append(class_name)
                    data.append(landmarks)

    # Write the data to a CSV file
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f'landmark_{i}' for i in range(len(data[0]) - 1)] + ['class']
        writer.writerow(header)
        writer.writerows(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filename to save the CSV file')
    parser.add_argument('filename', type=str, help='The name of the CSV file to save the landmarks to')

    args = parser.parse_args()

    main(f'{args.filename}.csv')
