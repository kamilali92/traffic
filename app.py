from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model import train_model
from src.data_preprocessing import load_data

app = Flask(__name__)

# Load the trained model (for simplicity, we retrain the model on app launch)
model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the user
        weather_condition = request.form['weather_condition']
        road_condition = request.form['Humidity(%)']
        temperature = float(request.form['temperature'])
        visibility = float(request.form['visibility'])
        
        # Prepare input data
        input_data = np.array([[weather_condition, road_condition, temperature, visibility]])
        
        # Standardize input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Predict accident severity
        prediction = model.predict(input_data_scaled)
        
        return render_template('index.html', prediction_text=f'Predicted Accident Severity: {prediction[0]}')
    except Exception as e:
        return str(e)

if __name__ == "__main__":

    app.run(debug=True)
