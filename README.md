# Iris-Classification
Iris Classification project using machine learning to classify iris flowers into Setosa, Versicolour, and Virginica species. Includes data exploration, model training with SVM, Logistic Regression, and Decision Tree, and performance evaluation. Contributions welcome! Python, Pandas, Scikit-learn, Seaborn, Matplotlib.
# Iris Classification Project

This project focuses on the classification of the Iris dataset using machine learning algorithms. The goal is to predict the species of iris flowers based on their features.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [Contact](#contact)

## Introduction

The Iris dataset is a classic dataset in the field of machine learning. This project uses Support Vector Machine, Logistic Regression, and Decision Tree algorithms to classify iris species.

## Installation

Clone the repository and install the required libraries:

''' bash
git clone https://github.com/yourusername/Iris-Classification.git
cd Iris-Classification
pip install -r requirements.txt

##  Usage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

## Load dataset
df = pd.read_csv("Iris.csv")
inputs = df.iloc[:, 0:4].values
targets = df.iloc[:, 4].values

## Split dataset
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=11)

## Train and evaluate models
model_svc = SVC().fit(x_train, y_train)
print("SVM Accuracy:", accuracy_score(y_test, model_svc.predict(x_test)) * 100)

model_logistic = LogisticRegression().fit(x_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, model_logistic.predict(x_test)) * 100)

model_tree = DecisionTreeClassifier().fit(x_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, model_tree.predict(x_test)) * 100)  

## Models and Evaluation

The project uses three models: SVM, Logistic Regression, and Decision Tree. Each model's accuracy is evaluated using the accuracy_score metric.
- Support Vector Machine (SVM): A supervised machine learning algorithm which can be used for both 
  classification or regression challenges.
- Logistic Regression: A statistical method for analyzing a dataset in which there are one or more 
  independent variables that determine an outcome.
- Decision Tree: A decision support tool that uses a tree-like model of decisions and their possible 
  consequences.
  
## Results
SVM Accuracy: 96.66%
Logistic Regression Accuracy: 100%
Decision Tree Accuracy: 100%

## Contact
For any inquiries, please contact your email.
https://github.com/Chandkund/Iris-Classification.git
