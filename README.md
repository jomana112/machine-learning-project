# Logistic-Regression-Decision-Tree-and-random-forest-

# Diabetes Prediction Using Machine Learning

This project uses various machine learning algorithms to predict whether a patient has diabetes based on the **Pima Indians Diabetes Dataset**. The models trained include Logistic Regression, Decision Trees, and Random Forest, evaluated with different sampling techniques like **Random Oversampling** and **SMOTE** (Synthetic Minority Over-sampling Technique). We will also use **GridSearchCV** to optimize model hyperparameters and visualize confusion matrices for model evaluation.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Methods and Algorithms](#methods-and-algorithms)
  - [Logistic Regression](#logistic-regression)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
- [Data Sampling Techniques](#data-sampling-techniques)
  - [Random Oversampling](#random-oversampling)
  - [SMOTE](#smote)
- [Model Evaluation](#model-evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Project Description

The aim of this project is to predict diabetes in patients based on medical and demographic data. This data was originally collected by the **National Institute of Diabetes and Digestive and Kidney Diseases**. The dataset contains 768 observations and 8 features such as age, BMI, glucose levels, and others.
In this project we build a Logistic Regression and Decision Tree model using Python and scikit-learn to solve a binary classification problem: predicting whether a person has diabetes or not based on medical data.
Working on the Pima Indians Diabetes Dataset, which contains several health-related attributes. The Goal is to apply the full machine learning pipeline, analyze the results, and understand how different factors like feature scaling, data splitting, and regularization impact model performance.

We evaluate the performance of several machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest

We apply **GridSearchCV** for hyperparameter tuning and evaluate the models on three datasets:
1. **Original Data**: Using the raw dataset without any resampling.
2. **Random Oversampling**: Synthetic oversampling of the minority class to balance the dataset.
3. **SMOTE**: Another resampling technique that generates synthetic samples for the minority class.

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Dataset**, available from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Installation

To run this project, you need to install the following Python libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
