import streamlit as st

st.set_page_config(
    page_title="Spam Prediction",
    page_icon="📧",
)

st.write("# Spam Prediction Model based on Online Learning")

st.markdown(
    """

Spam Filter is a Python project that uses a Support Vector Machine (SVM) model to classify emails as spam or non-spam (ham). This README provides an overview of the project, its functionality, and details about the Online SVM model used.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Online SVM Model](#online-svm-model)

## Introduction

The Spam Filter project is designed to classify emails into two categories: spam and non-spam. It uses a machine learning model, specifically an SVM classifier, to make predictions based on the content of the emails. This project demonstrates the following:

- Loading a pre-trained machine learning model.
- Processing and analyzing email data.
- Making predictions on new email content.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.

2. Ensure you have the required Python libraries installed. You can install them using `pip`:

```
   pip install pandas scikit-learn wordcloud matplotlib
```
3. Load the pre-trained SVM model using the provided model file.

4. Use the model to classify emails as spam or non-spam.

## Dataset

The project uses a dataset of labeled emails for training and testing the model. The dataset contains the following columns:

- **text**: The content of the email.
- **label_num**: The label, where 0 represents non-spam (ham) and 1 represents spam.
You can access the dataset on Kaggle by following this [link](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
## Online SVM Model

The Spam Filter utilizes an Online Support Vector Machine (SVM) model. This model is well-suited for scenarios where data arrives sequentially and must be processed incrementally. Key characteristics of the Online SVM model include:

- **Incremental learning**: The model can be updated with new data points without retraining on the entire dataset.

- **Hinge loss**: The model uses hinge loss as the loss function, which is commonly used in SVMs for binary classification.

- **Stochastic Gradient Descent (SGD)**: The model employs SGD for optimization, making it efficient for large datasets.

- **Real-time predictions**: The model can make real-time predictions as new email content is provided.

"""
)