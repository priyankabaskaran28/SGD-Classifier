# SGD-Classifier

## PRIYANKA B (212224060197)

## AIM:
To write a Python program to predict the type of species of the Iris flower using the SGD Classifier.

## EQUIPMENTS REQUIRED:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## ALGORITHM

1. Import the required Python libraries such as Pandas and Scikit-learn.

2. Load the Iris dataset containing sepal length, sepal width, petal length, and petal width.

3. Separate the four flower measurements as input features and the Iris species as the target variable.

4. Split the dataset into training and testing data.

5. Standardize the input features using `StandardScaler`.

6. Create an `SGDClassifier` model and train it using the training dataset.

7. Predict the Iris species for the testing dataset.

8. Calculate the accuracy of the SGD Classifier.

9. Display the actual and predicted species.

10. Predict the species of a new Iris flower using its sepal and petal measurements.

## PROGRAM:
~~~
Program to implement the prediction of Iris species using SGD Classifier.

Developed by: PRIYANKA B
RegisterNumber: 212224060197
~~~

```python
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
data["target"] = iris.target

# Display the first five records
print("First five records of Iris dataset:")
print(data.head())

# Input features
X = iris.data

# Target variable
y = iris.target

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Classifier
model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict Iris species
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display actual and predicted values
print("\nActual Species:")
print([iris.target_names[i] for i in y_test])

print("\nPredicted Species:")
print([iris.target_names[i] for i in y_pred])

# Display accuracy
print("\nAccuracy:", accuracy)

print("\nAccuracy Percentage:", accuracy * 100, "%")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict the species of a new Iris flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Standardize new flower data
new_flower_scaled = scaler.transform(new_flower)

# Predict species
prediction = model.predict(new_flower_scaled)

print("\nNew Flower Measurements:")
print("Sepal Length: 5.1 cm")
print("Sepal Width: 3.5 cm")
print("Petal Length: 1.4 cm")
print("Petal Width: 0.2 cm")

print(
    "\nPredicted Iris Species:",
    iris.target_names[prediction[0]]
)
```

## Output:

<img width="1076" height="778" alt="image" src="https://github.com/user-attachments/assets/4c8c1107-cbd1-4d49-b98d-53ca8ca6921d" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

