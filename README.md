# IMDB Movie Reviews Sentiment Analysis

This project focuses on sentiment analysis of IMDB movie reviews using deep learning techniques. It involves building a model to classify movie reviews as positive or negative based on the sentiment expressed in the text.

## Overview

Sentiment analysis, also known as opinion mining, aims to determine the sentiment expressed in a piece of text. In this project, we use a dataset consisting of 50,000 IMDB movie reviews labeled as positive or negative sentiment.

## Dataset

The dataset used in this project is publicly available on Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Requirements

To run the code, the following libraries are required:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- tensorflow
- joblib

You can install these libraries using pip:

```
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow joblib
```

## Instructions

1. Clone the repository to your local machine.
2. Make sure you have installed all the required libraries.
3. Run the provided Jupyter Notebook or Python script in your preferred environment.
4. Follow the code execution step by step to understand the process of sentiment analysis.
5. Make modifications or enhancements as needed for your specific use case.

## Code Structure

- **Data Loading and Preprocessing**: Includes loading the dataset, visualizing data distribution, and preprocessing steps such as removing stopwords.
- **Tokenization and Padding**: Tokenizes the text data and pads sequences to a fixed length for model input.
- **Model Architecture**: Defines the deep learning model architecture using TensorFlow's Keras API.
- **Training the Model**: Trains the model on the training data and visualizes training/validation loss and accuracy.
- **Model Evaluation**: Evaluates the trained model on the testing data and generates a classification report and confusion matrix.
- **Saving the Model**: Saves the trained model using joblib for future use.

## Results

After training the model, the accuracy achieved on the testing data is displayed along with a classification report and confusion matrix, providing insights into the model's performance.

## Conclusion

Sentiment analysis is a valuable tool for understanding public opinion and can be applied in various domains such as product reviews, social media sentiment analysis, and customer feedback analysis. This project demonstrates the implementation of sentiment analysis using deep learning techniques on IMDB movie reviews dataset.
