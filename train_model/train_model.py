import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("isl_alphabets_webcam.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
with open("isl_knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
