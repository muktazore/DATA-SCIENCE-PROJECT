import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Step 1: Data Collection and Preprocessing
# Example: Using a sample dataset (Iris dataset)
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")

# Step 3: API Development using FastAPI
app = FastAPI()

# Define a request body model
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get('/')
def read_root():
    return {"message": "Welcome to the Iris Prediction API!"}

@app.post('/predict')
def predict(data: PredictionRequest):
    # Load the model
    model = joblib.load('model.pkl')

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([{  
        'sepal length (cm)': data.sepal_length,
        'sepal width (cm)': data.sepal_width,
        'petal length (cm)': data.petal_length,
        'petal width (cm)': data.petal_width
    }])

    # Make a prediction
    prediction = model.predict(input_data)[0]
    class_name = data.target_names[prediction]

    return {"prediction": prediction, "class": class_name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
