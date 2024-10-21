# DL ASL Recognition

## ASL Alphabet
The American Sign Language (ASL) is a language that uses hand gestures to communicate. The ASL alphabet is a set of 26 hand gestures that represent the 26 letters of the English alphabet. This project aims to recognize the ASL alphabet using deep learning.

<img src="imgs/ASL_ALPHABET.png">
 
## How to run?
### Before you start
Make sure you:
1. Have a webcam enabled device
2. Downloaded the images from the [Kaggle dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and placed them in the `src/data` directory if you are planning on training the model.
### Running the code
1. Clone the repository
2. Access the directory
3. Run the following command
```bash
pip install -r requirements.txt
```
### Workflow
1. Since the images dataset is too large, you need to start by capturing your own images. Run the following command to start capturing images.
```bash
python src/collect_data.py
```
2. After capturing the images, you can generate a CSV file with the landmark and class information. Run the following command to generate the CSV file.
```bash
python src/create_csv.py
```
