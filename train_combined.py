import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load both datasets
alphabets = pd.read_csv("isl_alphabets_webcam.csv", header=None)
numbers = pd.read_csv("isl_numbers_webcam.csv", header=None)

# Combine datasets
combined = pd.concat([alphabets, numbers], ignore_index=True)

X = combined.iloc[:, :-1].values

#Force labels to string
y = combined.iloc[:, -1].astype(str).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Combined Model Accuracy:", accuracy)

# Save model
with open("isl_combined_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Combined model saved successfully.")
