Google Colab Link: https://colab.research.google.com/drive/1sbmaf_lcqHHBm19kPshIAIl16qdEaYZo#scrollTo=cpfSyszpyZJX

# Support-Vector-Machine-SVM-
Diabetes Prediction using Support Vector Machine (SVM)


# Diabetes Prediction using Support Vector Machine (SVM)

This project predicts whether a patient has diabetes using the **PIMA Diabetes dataset**. A **Support Vector Machine (SVM)** classifier is applied to classify patients based on their health metrics. The steps include data collection, preprocessing, model training, and making predictions.

---

## Table of Contents
1. [Tools and Libraries](#tools-and-libraries)
2. [Importing Dependencies](#importing-dependencies)
3. [Data Collection and Analysis](#data-collection-and-analysis)
4. [Labeling Features and Target](#labeling-features-and-target)
5. [Data Standardization](#data-standardization)
6. [Splitting the Dataset](#splitting-the-dataset)
7. [Training the SVM Classifier](#training-the-svm-classifier)
8. [Model Evaluation](#model-evaluation)
9. [Predictive System](#predictive-system)
10. [Q&A](#qa)
11. [Conclusion](#conclusion)

---

## Tools and Libraries

This project uses several Python libraries for data processing, machine learning, and evaluation.

- **NumPy**: Provides support for working with arrays and matrices.
- **Pandas**: Useful for handling and manipulating tabular data (data frames).
- **StandardScaler (from sklearn.preprocessing)**: Standardizes features by removing the mean and scaling to unit variance, making the data ready for machine learning models.
- **Train_test_split (from sklearn.model_selection)**: Splits the dataset into training and testing sets to evaluate model performance.
- **SVM (from sklearn)**: The support vector machine algorithm is used for classification tasks. Here, it helps classify diabetes patients.
- **Accuracy_score (from sklearn.metrics)**: Used to measure the accuracy of the model by comparing the predicted values with actual outcomes.

---

## Importing Dependencies

```python
import numpy as np  # For array operations
import pandas as pd  # For handling tabular data
from sklearn.preprocessing import StandardScaler  # For data standardization
from sklearn.model_selection import train_test_split  # Splitting the dataset
from sklearn import svm  # Support Vector Machine classifier
from sklearn.metrics import accuracy_score  # Evaluating accuracy
```

---

## Data Collection and Analysis

The dataset is sourced from the **PIMA Indians Diabetes Database**. It contains diagnostic measurements related to diabetes.

```python
# Loading the dataset
diabetes_data = pd.read_csv('/content/diabetes_data_set.csv')

# Displaying the first 5 rows
diabetes_data.head()

# Checking the structure of the dataset
diabetes_data.shape

# Summary statistics
diabetes_data.describe()

# Checking the distribution of 'Outcome'
diabetes_data['Outcome'].value_counts()
```

---

## Labeling Features and Target

Here, we separate the dataset into features (X) and target (Y), where the target is the outcome (whether or not the patient has diabetes).

```python
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

print(X)
print(Y)
```

---

## Data Standardization

Standardizing the features so that all variables contribute equally to the model:

```python
scaler = StandardScaler()

# Fitting and transforming the feature data
scaler.fit(X)
standardized_data = scaler.transform(X)

print(standardized_data)
```

---

## Splitting the Dataset

Splitting the dataset into training (80%) and testing (20%) sets for model evaluation:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape, Y.shape, Y_train.shape, Y_test.shape)
```

---

## Training the SVM Classifier

Training an SVM model using a linear kernel to classify the data.

```python
classifier = svm.SVC(kernel='linear')

# Training the model
classifier.fit(X_train, Y_train)
```

---

## Model Evaluation

Evaluating the model's performance on both training and testing data:

```python
# Predicting on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Training data accuracy:', training_data_accuracy)

# Predicting on testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Test data accuracy:', test_data_accuracy)
```

---

## Predictive System

Building a predictive system to classify whether a new patient is diabetic based on their health metrics.

```python
input_data = (8, 183, 64, 0, 0, 23.3, 0.672, 32)  # Example data
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the input data for standardization
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

# Making a prediction
prediction = classifier.predict(std_data)
print(prediction)

if prediction == 1:
    print('Patient has diabetes')
else:
    print('Patient does not have diabetes')
```

---

## Q&A

1. **Why is standardization important in this project?**
   - Standardization ensures that all features contribute equally to the model, avoiding bias toward features with larger values.

2. **Why use SVM for classification?**
   - SVM is a powerful algorithm for binary classification problems. It works well with both linear and non-linear datasets and is especially useful when the data is not linearly separable.

3. **What is the role of train_test_split?**
   - The `train_test_split` function ensures that the model is evaluated on unseen data (test set) to avoid overfitting and to check the generalization performance of the model.

4. **How does accuracy score help?**
   - The accuracy score compares the predicted values with the actual values, giving a percentage of correct predictions made by the model.

---

## Conclusion

In this project, we successfully built a model to predict diabetes using the PIMA dataset. The **SVM classifier** performed well, providing a good accuracy score for both training and testing datasets. Data preprocessing like **standardization** and proper **train-test split** were key steps to achieving this result. This predictive model can now be used to classify new patients based on their health metrics and assess their risk of having diabetes.

---

This comprehensive structure should help present your project on GitHub in a clear and professional manner.
