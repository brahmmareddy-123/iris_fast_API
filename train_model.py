import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("Iris.csv")


# Drop Id
df = df.drop("Id", axis=1)


# Features / target
X = df.drop("Species", axis=1)
y = df["Species"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Save model
joblib.dump(model, "iris_model.pkl")

print("Model trained successfully")