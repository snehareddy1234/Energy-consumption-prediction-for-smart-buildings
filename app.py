import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the data
data = pd.read_csv('C:/Users/sneha/Desktop/Energy consumption project/data/Energy_consumption_with_HomeID.csv')

# Preprocess the data
data = pd.get_dummies(data, columns=['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday'], drop_first=True)
X = data.drop(['EnergyConsumption', 'Timestamp', 'HomeID'], axis=1)
y = data['EnergyConsumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'models/model.pkl')

# Calculate accuracy metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
accuracy = 100 - (mae / np.mean(y_test) * 100)  

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = request.form.to_dict()
    
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode the categorical variables
    input_df = pd.get_dummies(input_df, columns=['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday'], drop_first=True)
    
    # Ensure the input DataFrame has the same columns as the training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    return render_template('index.html', prediction_text=f'Predicted Energy Consumption for Home ID {input_data["HomeID"]}: {prediction[0]:.2f}, Model Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)

    