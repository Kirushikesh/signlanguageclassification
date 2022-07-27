# Sign Language Prediction
## Table of Content
- [Demo](#demo)
- [Screen Shots](#screen-shots)
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [Run](#run)
- [Technologies and Tools](#technologies-and-tools)
- [To Do](#to-do)
- [Contact](#contact)
## Demo
## Screen Shots

![image](https://user-images.githubusercontent.com/49152921/117792695-26f78d00-b269-11eb-9689-5cdfa8290f5f.png)

## Overview
Sign Language Classification using live video feed from the camera. This project is used to identify the english alphabets using corresponding sign languages. Where J and Z are the outliers which cannot be found using this project due to their gesture motions. Classical CNN model helps us to identify the alphabet from the sign, the model was build with the accuracy of 97%, the dataset used for the problem is [sign language mnist dataset from kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).
## Motivation
Communication is not reserved for hearing people alone, and using one's voice is not the only way to communicate. Deaf and Dumb use various ways for communicating which among one is sign language. Even through they learn the sign language in their early days, many of us dont know it because the place we study wont teach us that. Usually there will be a translator who will translate the sign language. This project was build on this idea to remove the translator, and fills the gap between them and us.
## Technical Aspect
This project was divided into two parts
1. [Training a Convolutional model using Keras.](https://github.com/Kirushikesh/signlanguageclassification/blob/main/SignlanguageModel.ipynb)
2. [Using that model to identify the sign language efficiently.](https://github.com/Kirushikesh/signlanguageclassification/blob/main/prediction.py)
## Installation
The Code is written in Python 3.7. If you don't have Python installed you should install it first. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```python
pip install -r requirements.txt
```
## Run
Before that you first get your kaggle api token, if you are new to this follow this [link](https://www.kaggle.com/docs/api) to get yours it will be useful for directly using kaggle dataset in colab.
- Run ```SignLanguageModel.ipynb``` which will ask you to upload your kaggle.json file(api token) at the end you will get your cnn model as mycnn.h5 .
- Save the CNN model in the same folder of our prediction.py file.
- Run ```prediction.py``` which uses the live feed from your camera and classify the sign languages. Place your camera opposite to plain wall for better prediction. Initially wait 4 to 5 seconds for the program to fetch the background which will be useful for seperating the objects(your hand) from the background. After that make sure to use signs with in the Region of Interest. Thats all now the identified alphabets will appear in another window.

## Technologies and Tools
- Python 
- Tensorflow
- Keras
- OpenCV

## To Do
1. Increase the vocabulary of our model.
2. Deploy the project on cloud and create an API for using it.

## Contact
If you found any bug or like to raise a question feel free to contact me through [LinkedIn](https://www.linkedin.com/in/kirushikesh-d-b-10a75a169/).
If you feel this project helped you and like to encourage me for more stuffs like this please endorse my skills in my [LinkedIn](https://www.linkedin.com/in/kirushikesh-d-b-10a75a169/) thanks in advance!!!.
