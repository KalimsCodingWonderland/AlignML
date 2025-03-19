from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pymongo import MongoClient
import os
import traceback

app = Flask(__name__)

# Connect to MongoDB. Set your MONGO_URI environment variable accordingly.
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://kalimqazi05:sLCrNILXU06cfbhK@cluster0.higw9.mongodb.net?retryWrites=true&w=majority&appName=Cluster0')
client = MongoClient(MONGO_URI)
db = client.get_default_database()  # or client['your_db_name']

def get_training_data(user_id):
    """
    Query the MongoDB 'feedback' collection for training data.
    Each document should have: user_id, category, and duration (in minutes).
    """
    cursor = db.feedback.find({"user_id": user_id})
    data = list(cursor)
    if not data:
        return None
    df = pd.DataFrame(data)
    # Ensure required columns exist.
    if "category" not in df.columns or "duration" not in df.columns:
        return None
    return df

def train_model(user_id):
    df = get_training_data(user_id)
    if df is None or len(df) < 5:
        return None  # Not enough data to train
    # One-hot encode the 'category' field.
    X = pd.get_dummies(df["category"])
    y = df["duration"]
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
            # Fallback to 30 minutes if not enough training data.
            return jsonify({"predicted_duration": 30})
        model, columns = model_data
        # Prepare the input data.
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
        user_duration = data.get("user_duration")
        # Insert new feedback into the 'feedback' collection.
        db.feedback.insert_one({
            "user_id": user_id,
            "category": category,
            "duration": user_duration
        })
        return jsonify({"status": "success"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
