import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("dataset/irrigation_data.csv")

print("Dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# --------------------------------
# 2. Encode Plant
# --------------------------------

encoder = LabelEncoder()

df["Plant"] = encoder.fit_transform(df["Plant"])


# --------------------------------
# 3. Separate input and output
# --------------------------------

X = df[
    [
        "Plant",
        "SoilMoisture",
        "Temperature",
        "Humidity",
        "Rain",
        "WaterLevel"
    ]
]

y = df["Irrigation"]


# --------------------------------
# 4. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 5. Create Random Forest model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------
# 6. Train model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 7. Test model
# --------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 8. Save model
# --------------------------------

joblib.dump(model, "irrigation_model.pkl")

joblib.dump(encoder, "plant_encoder.pkl")

print("\nModel saved successfully!")
print("irrigation_model.pkl")
print("plant_encoder.pkl")