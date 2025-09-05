# Breast-Cancer-Classification-NN-KNN
## Overview
This project implements **Breast Cancer Classification** using **Neural Networks (NN)** and **k-Nearest Neighbors (k-NN)** in R.  
It uses the `BreastCancer` dataset from the `mlbench` package to predict whether a tumor is benign or malignant.

## Features
- Data preprocessing (handling missing values, encoding labels)
- Neural Network model using `deepnet`
- K-Nearest Neighbors model using `class` package
- Evaluation metrics (accuracy, confusion matrix)
- Optional: Deep learning model using `keras` for advanced predictions

## Requirements
- R (>= 4.0)
- Packages: `keras`, `deepnet`, `mlbench`, `caret`, `class`

You can install the packages with:
```R
install.packages(c("keras", "deepnet", "mlbench", "caret", "class"))
