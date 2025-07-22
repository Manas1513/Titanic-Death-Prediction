# Titanic-Death-Prediction

Titanic Survival Prediction
Project Overview
This project aims to predict whether a passenger on the RMS Titanic survived or not based on various features available in the dataset. The analysis involves data cleaning, exploratory data analysis, feature engineering, and building a predictive model using the K-Nearest Neighbors (KNN) algorithm.

Dataset
The project uses the "Titanic-Dataset.csv" file, which contains passenger information such as age, class, sex, and fare, along with the survival outcome.

Dataset Columns:

PassengerId: Unique ID for each passenger.

Survived: Survival status (0 = No, 1 = Yes).

Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).

Name: Passenger's name.

Sex: Passenger's sex.

Age: Passenger's age in years.

SibSp: Number of siblings/spouses aboard the Titanic.

Parch: Number of parents/children aboard the Titanic.

Ticket: Ticket number.

Fare: Passenger fare.

Cabin: Cabin number.

Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Methodology
The analysis follows these key steps:

Data Loading & Initial Exploration: The dataset is loaded using pandas, and an initial inspection is performed to understand its structure and identify missing values.

Data Cleaning & Preprocessing:

Missing values in Survived, Age, Cabin, and Fare are handled. The Cabin column is dropped due to a high number of missing values.

Forward-fill (ffill) is used to impute missing data.

Outliers in numerical features (Fare, Age, SibSp, Pclass) are identified using the Interquartile Range (IQR) method and are capped to reduce their skewing effect on the model.

Feature Selection: The following features were selected as predictors (X) for the model:

PassengerId

Pclass

Age

SibSp

Fare

Parch
The Survived column is the target variable (y).

Model Training & Evaluation:

The data is split into training (80%) and testing (20%) sets.

Features are scaled using MinMaxScaler to normalize the data and ensure that all features contribute equally to the model's performance.

A K-Nearest Neighbors (KNN) classifier is trained on the scaled training data.

Hyperparameter Tuning:

To find the optimal number of neighbors (k), the model was trained and evaluated with k values ranging from 1 to 49.

The training and testing accuracies were plotted against the different k values to visualize the model's performance and select a k that generalizes well to unseen data. Based on the analysis, a k of 37 was chosen.

Results
The final KNN model with n_neighbors=37 achieved the following accuracy scores:

Training Accuracy: 71.76%

Test Accuracy: 71.51%

The similar performance on both training and testing sets suggests that the model is not overfitting and generalizes well.

Requirements
To run this project, you will need Python 3 and the following libraries:

pandas

numpy

matplotlib

scikit-learn

seaborn

You can install these dependencies using pip:

pip install pandas numpy matplotlib scikit-learn seaborn

Usage
Clone this repository to your local machine.

Make sure you have the Titanic-Dataset.csv file in the same directory as the script.

Run the Python script:

python Titanic.py

The script will execute the data analysis and print the final model accuracy scores to the console.

License
This project is licensed under the MIT License. See the LICENSE file for details.

