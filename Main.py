import pandas as pd

from SVM_Model import train_svm_model, predict_emotion
from connectGPT_API import generate_prompt, get_gpt_response
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning)

# Train the SVM model
train_data_path = 'Train.csv'
svm, scaler, location_encoder = train_svm_model(train_data_path)

# Load new data testing data
new_data_path = "Test.csv"  # Replace with your new data file path
new_data = pd.read_csv(new_data_path)

# Convert 'DateTime' column to datetime objects
new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format="%d/%m/%Y %H:%M")

# Encode 'Location' variable
new_data['Location'] = location_encoder.transform(new_data['Location'])

# Set DateTime as the index
new_data.set_index('DateTime', inplace=True)

previous_emotion = None
# predict the emotion using the trained model with predict function
for day_index in range(len(new_data)):
    row = new_data.iloc[[day_index]]
    X_new = scaler.transform(row[['HR', 'Weather', 'Skin_Temperature', 'Location']].values.reshape(1, -1))
    predicted_emotion = svm.predict(X_new)

# check if previous emotion is not the same as the previous one if not send a prompt to the gpt API
    if predicted_emotion[0] != previous_emotion:
        prompt = generate_prompt(predicted_emotion[0])
        quote = get_gpt_response(prompt)
# print the emotion and the quotes in the terminal
        print(f"Predicted emotion: {predicted_emotion[0]}")
        print(f"Motivational quote: {quote}")
        print()

    previous_emotion = predicted_emotion[0]
