import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from EmotionFunc import detect_emotion


# Function to be used in the testing data to predict emotions
def predict_emotion(svm_model, scaler, location_encoder, new_data):
    # Preprocess the new data by encoding the location and formatting the date 
    new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format="%d/%m/%Y %H:%M")
    new_data['Location'] = location_encoder.transform(new_data['Location'])
    new_data.set_index('DateTime', inplace=True)

    # Group the data by 24-hour 
    grouped_new_data = new_data.groupby(pd.Grouper(freq='24H')).agg({
        'HR': 'mean',
        'Weather': 'mean',
        'Skin_Temperature': 'mean',
        'Location': lambda x: x.mode().iat[0]
    })

    # Reset the index
    grouped_new_data.reset_index(inplace=True)

    # Select features to predict the emotion the model
    X_new = grouped_new_data[['HR', 'Weather', 'Skin_Temperature', 'Location']]
    X_new_scaled = scaler.transform(X_new)

    # Use the model to predict the emotion
    predictedEmotion = svm_model.predict(X_new_scaled)

    return predictedEmotion


# function to train the svm model on the training data
def train_svm_model(train_data_path):
    # Load the data
    train_data = pd.read_csv(train_data_path)

    # Convert 'DateTime' column to datetime objects
    train_data['DateTime'] = pd.to_datetime(train_data['DateTime'], format="%d/%m/%Y %H:%M")

    # Encode categorical variables
    location_encoder = LabelEncoder()
    train_data['Location'] = location_encoder.fit_transform(train_data['Location'])

    # Set DateTime as the index
    train_data.set_index('DateTime', inplace=True)

    # Group the data by 24-hour intervals
    grouped_data = train_data.groupby(pd.Grouper(freq='24H')).agg({
        'HR': 'mean',
        'Weather': 'mean',
        'Skin_Temperature': 'mean',
        'Location': lambda x: x.mode().iat[0]
    })

    # Reset the index
    grouped_data.reset_index(inplace=True)

    # Assign emotion labels to all groups in the data
    grouped_data['Emotion'] = detect_emotion(train_data_path)

    # Split the data into features (X) and target (y) variables
    X = grouped_data[['HR', 'Weather', 'Skin_Temperature', 'Location']]
    y = grouped_data['Emotion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = svm.predict(X_test_scaled)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')  # print the accuracy of the model

    return svm, scaler, location_encoder


# print run the model using the traing and the testing data.
if __name__ == "__main__":
    train_data_path = "Train.csv" # training data for the svm model
    svm_model, scaler, location_encoder = train_svm_model(train_data_path)

    # Load the testing data file
    new_data_path = "Test.csv"
    new_data = pd.read_csv(new_data_path)

    # Predict emotion and using the predicted_emotion function
    predicted_emotion = predict_emotion(svm_model, scaler, location_encoder, new_data)
    print("Predicted emotion:", predicted_emotion)
