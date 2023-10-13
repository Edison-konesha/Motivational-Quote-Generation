from flask import Flask, jsonify, render_template, send_from_directory
from SVM_Model import train_svm_model, predict_emotion
from connectGPT_API import generate_prompt, get_gpt_response
import pandas as pd
import threading
import warnings
from sklearn.exceptions import DataConversionWarning

# ignore any warnings that come up from the sklearn package
warnings.filterwarnings(action='ignore', category=UserWarning)

app = Flask(__name__, static_folder='static')


# Flask static path to find the route of the emotion images related to the emotion predicted
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


# Train the SVM model
train_data_path = 'Train.csv'
svm, scaler, location_encoder = train_svm_model(train_data_path)

# Load new data
new_data_path = "Test.csv"  # Replace with your new data file path
new_data = pd.read_csv(new_data_path)

# Convert 'DateTime' column to datetime objects
new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format="%d/%m/%Y %H:%M")

# Encode 'Location' variable
new_data['Location'] = location_encoder.transform(new_data['Location'])

# Set DateTime as the index
new_data.set_index('DateTime', inplace=True)

# store the generated quotes in an array so that they can be accessed by the browser
quotes = []
previous_emotion = None


# predicts the new ques using the trained svm model based on the training data
# function is also used on the front end of the flask application
def generate_quotes():
    global quotes
    global previous_emotion

    for day_index in range(len(new_data)):
        row = new_data.iloc[day_index]
        X_new = scaler.transform(row[['HR', 'Weather', 'Skin_Temperature', 'Location']].values.reshape(1, -1))
        predicted_emotion = svm.predict(X_new)

        if predicted_emotion[0] != previous_emotion:
            prompt = generate_prompt(predicted_emotion[0])
            quote = get_gpt_response(prompt)
            quotes.append({
                'day': day_index + 1,
                'predicted_emotion': predicted_emotion[0],
                'motivational_quote': quote
            })

        previous_emotion = predicted_emotion[0]


# Start generating quotes in a separate thread so that generation can continue
# while the browser is also running
quote_thread = threading.Thread(target=generate_quotes)
quote_thread.start()


# send the quote to the front end to display in the browser
@app.route('/get_quote/<int:day_index>', methods=['GET'])
def get_quote(day_index):
    if day_index < 0 or day_index >= len(quotes):
        return jsonify({"error": "Invalid day index"}), 400

    return jsonify(quotes[day_index])


# use the render_template function to look for the html using the flask notation
@app.route('/')
def home():
    return render_template('index.html')


# start the debugger when the main app is running
if __name__ == "__main__":
    app.run(debug=True)
