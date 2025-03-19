from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import traceback

app = Flask(__name__)
DATA_FILE = "historical_data.csv"

# Load historical data if exists; otherwise create an empty DataFrame.
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["user_id", "category", "duration"])  # duration in minutes

def train_model(user_id):
    user_data = df[df["user_id"] == user_id]
    if len(user_data) < 5:
        return None  # Not enough data to train
    X = pd.get_dummies(user_data["category"])
    y = user_data["duration"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        category = data.get("category")
        model_data = train_model(user_id)
        if model_data is None:
            # Fallback to 30 minutes if not enough data.
            return jsonify({"predicted_duration": 30})
        model, columns = model_data
        # Prepare input for prediction.
        input_df = pd.DataFrame([[category]], columns=["category"])
        input_df = pd.get_dummies(input_df["category"])
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[columns]
        prediction = model.predict(input_df)[0]
        predicted_duration = max(1, int(round(prediction)))
        return jsonify({"predicted_duration": predicted_duration})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"predicted_duration": 30})

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        category = data.get("category")
        predicted_duration = data.get("predicted_duration")
        user_duration = data.get("user_duration")
        global df
        new_entry = {"user_id": user_id, "category": category, "duration": user_duration}
        df = df.append(new_entry, ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        return jsonify({"status": "success"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error"})

if __name__ == '__main__':
    app.run(port=5002)
