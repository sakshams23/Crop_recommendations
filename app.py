import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

# Create Flask app
app = Flask(__name__)

# Load your trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting the form data
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Applying scalers: MinMaxScaler then StandardScaler
    scaled_features = ms.transform(final_features)
    final_scaled_features = sc.transform(scaled_features)
    
    # Making predictions
    prediction = model.predict(final_scaled_features)
    
    # Dictionary to map the predicted class to a crop
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
        15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    
    # Get the predicted crop
    predicted_crop = crop_dict.get(prediction[0], "Unknown crop")

    # Return result to the same page
    return render_template('index.html', prediction_text="Best crop for the input is: {}".format(predicted_crop))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    
    # Convert data into a numpy array
    features = [np.array(list(data.values()))]
    
    # Apply scalers
    scaled_features = ms.transform(features)
    final_scaled_features = sc.transform(scaled_features)
    
    # Predict using model
    prediction = model.predict(final_scaled_features)
    
    # Return prediction result
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
