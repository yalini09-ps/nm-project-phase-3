import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, jsonify

# Load dataset (using a public diabetes dataset as an example)
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)
    return data

# Preprocess dataset
def preprocess_data(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Evaluation
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

# Save model
def save_model(clf, filename="disease_predictor.pkl"):
    joblib.dump(clf, filename)

# Load model
def load_model(filename="disease_predictor.pkl"):
    return joblib.load(filename)

# Flask API for prediction
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload with keys:
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    """
    data = request.get_json(force=True)
    features = [[
        data.get("Pregnancies", 0),
        data.get("Glucose", 0),
        data.get("BloodPressure", 0),
        data.get("SkinThickness", 0),
        data.get("Insulin", 0),
        data.get("BMI", 0),
        data.get("DiabetesPedigreeFunction", 0),
        data.get("Age", 0)
    ]]
    clf = load_model()
    prediction = clf.predict(features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    # Train and save model if not already done
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    save_model(clf)
    # Uncomment the next line to run the Flask API
    # app.run(debug=True)