# Machine Learning Portfolio

## Introduction
  As an aspiring and experienced Machine Learning Engineer, wanted to build a personal Machine Learning portfolio to share learnings, and experiences and collaborate with people who have the same interest in this field.

## Projects

## Deep Learning
### Face Detector or Face Recognition
>**Keywords:** numpy, matplotlib.pyplot, pathlib, PIL(Image, ImageDraw) and face_recognition *[View Source](https://github.com/rekha0suthar/machine-learning-portfolio/tree/main/face_detector)*

In this project, I have used the **face_recognition** library. With the help of this library, I get face location, face embeddings, and many more features from Image. This project can detect faces in images, draw rectangles, or lines over facial features(nose, eyebrows, lips, etc.) and do some digital makeup. At the end we are giving input image and the model is giving a similar face from training data.

### Digit Recognition
>**Keywords:** numpy, pandas, Tensorflow, Keras, gradio *[View Source](https://github.com/rekha0suthar/machine-learning-portfolio/tree/main/digit-recognizer)*

This is a simple **Convolutional Neural Network(CNN)** model. I have used Tensorflow and Keras in this project. I have added some **Convolutional** layers, some **Max Pooling** layers, and a **Dense** layer, and also applied **Flatten**, **Dropout**, and **RELU** activation functions. In the end, using gradio built UI where user can draw number and output will be shown on the right side with the top three highest probability digits with percentages.

### Image Classification - Cat-Dog Classifier
>**Keywords:** fastai, gradio *[View Source](https://github.com/rekha0suthar/deep-learning-projects/tree/main/cat-dog-classifier)*

  I have built this model with the help of **fastai**  library and built User Interface using Gradio. We give input image of a dog or cat and the model gives output with either it is cat or dog with percentage. In UI, we can upload an image and the output will be shown on the right side. I have deployed this model on Hugging Face.

## Machine Learning
### Price Prediction of House - Regression
> **Keywords:** numpy, pandas, matplotlib.pyplot, seaborn, scikit-learn *[View Source](https://github.com/rekha0suthar/boston-house-pricing-regmodel)*

  This is a simple Traditional Machine Learning Project. I have done Data Proprocessing, Exploratory Data Analysis(EDA), and Data Scaling, and in the end, used **LinearRegression** Model to train data and make predictions. I have built **HTML** page for UI purposes and getting input data from the user through form and sending that data to **POST API** to see prediction using POSTMAN. Using the model to predict output and then showing it on HTML Page. I have also added a **.yaml** file and **Procfile** to deploy it on Heroku. Also added the **Docter** file.

### Wine Quality Prediction - Classification
> **Keywords:** numpy, pandas, matplotlib.pyplot, seaborn, scikit-learn *[View Source](https://github.com/rekha0suthar/machine-learning-projects/tree/main/Wine%20Quality%20Prediction)*

This is a simple Classification Model to predict the Quality of Wine. I have done Data Proprocessing, Exploratory Data Analysis(EDA), and Data Balancing using **SMOTE**, and finally tried many classification Models like LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and ExtraTreeClassifier. Among all these **ExtraTreeClassifier** gave the highest accuracy.
