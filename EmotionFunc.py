import pandas as pd
import random


# runs all the functions for the selected emotions
# selects an emotion randomly based on a 24-hour period
# returns an array of detected emotions
def detect_emotion(dataFile):
    # Load the data from the file
    data = pd.read_csv(dataFile)

    # Convert the date and time column to match the correct format
    data['DateTime'] = pd.to_datetime(data['DateTime'], format="%d/%m/%Y %H:%M")

    # Set the DateTime column as the DataFrame index
    data.set_index('DateTime', inplace=True)

    # Group the data by 24-hour 
    grouped_data = data.groupby(pd.Grouper(freq='24H'))

    emotions = [
        emotion_happy,
        emotion_sad,
        emotion_angry,
        emotion_scared,
        emotion_anxiety,
        emotion_guilt,
        emotion_loneliness,
        emotion_panicked,
    ]

    predicted_emotion = []

    for _, dayData in grouped_data:
        candidate_emotions = []

        for emotion_func in emotions:
            emotion, true_conditions = emotion_func(dayData)
            if emotion is not None:
                candidate_emotions.append(emotion)

        if candidate_emotions:
            day_emotion = random.choice(candidate_emotions)
        else:
            day_emotion = "Unknown"

        predicted_emotion.append(day_emotion)

    return predicted_emotion


# Happy emotion when a data row matches its constraints
def emotion_happy(dayData):
    conditions = [
        (60 < dayData['HR'].mean() < 80) or (80 < dayData['HR'].mean() < 100),
        35 < dayData['Skin_Temperature'].mean() < 37 or
        20 < dayData['Weather'].mean() < 30,
        dayData['Location'].str.contains(
            'Home|Park|Outdoor Recreation Area|Friend\'s or Family Member\'s House|Place of Worship').any(),
        dayData['condition'].str.contains('NoStress').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 1  # Set the threshold for the number of conditions that need to be true for the emotion to be detected

    if true_conditions >= threshold:
        return 'Happy', true_conditions

    return None, 0


# sad emotion when a data row matches its constraints
def emotion_sad(dayData):
    conditions = [
        dayData['HR'].mean() > 40 and dayData['HR'].mean() < 70,
        dayData['Skin_Temperature'].mean() > 35 and dayData['Skin_Temperature'].mean() < 37,
        (dayData['Weather'].mean() > 10 or dayData['Weather'].mean() < 20) or dayData['Weather'].mean() < 0,
        dayData['Location'].str.contains('Home|Library or Community Center|Place of Worship').any(),
        dayData['condition'].str.contains('interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 4

    if true_conditions >= threshold:
        return 'Sad', true_conditions

    return None, 0


# Angry emotion when a data row matches its constraints
def emotion_angry(dayData):
    conditions = [
        dayData['HR'].mean() > 80 and dayData['HR'].mean() < 100,
        dayData['Skin_Temperature'].mean() > 36 or dayData['Skin_Temperature'].mean() < 38,
        dayData['Location'].str.contains('Work|Public Transportation|Shopping Mall|Retail Store').any() or
        dayData['condition'].str.contains('Interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 3

    if true_conditions >= threshold:
        return 'Angry', true_conditions

    return None, 0


# Scared emotion when a data row matches its constraints
def emotion_scared(dayData):
    conditions = [
        dayData['HR'].mean() > 90 and dayData['HR'].mean() < 110,
        dayData['Skin_Temperature'].mean() > 33 and dayData['Skin_Temperature'].mean() < 35,
        dayData['Location'].str.contains(
            'Movie Theater|Entertainment Venue|Outside Activity|Unfamiliar Environment').any() or
        dayData['condition'].str.contains('Interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 3

    if true_conditions >= threshold:
        return 'Scared', true_conditions

    return None, 0


# Anxiety emotion when a data row matches its constraints
def emotion_anxiety(dayData):
    conditions = [
        dayData['HR'].mean() > 80 and dayData['HR'].mean() < 100,
        dayData['Skin_Temperature'].mean() > 35 or dayData['Skin_Temperature'].mean() < 37,
        dayData['Location'].str.contains(
            'Work|Medical or Dental Office|Public Transportation|Unfamiliar Environment').any(),
        dayData['condition'].str.contains('Interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 1

    if true_conditions >= threshold:
        return 'Anxiety', true_conditions

    return None, 0


# Guilt emotion when a data row matches its constraints
def emotion_guilt(dayData):
    conditions = [
        dayData['HR'].mean() > 70 and dayData['HR'].mean() < 90,
        dayData['Skin_Temperature'].mean() > 35 and dayData['Skin_Temperature'].mean() < 37,
        dayData['Location'].str.contains('Home|Place of Worship|Library or Community Center').any(),
        dayData['condition'].str.contains('interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 2

    if true_conditions >= threshold:
        return 'Guilt', true_conditions

    return None, 0


# Loneliness emotion when a data row matches its constraints
def emotion_loneliness(dayData):
    conditions = [
        dayData['HR'].mean() > 50 and dayData['HR'].mean() < 70,
        dayData['Skin_Temperature'].mean() > 34 and dayData['Skin_Temperature'].mean() < 36,
        (dayData['Weather'].mean() > 10 or dayData['Weather'].mean() < 20) or dayData['Weather'].mean() < 0,
        dayData['Location'].str.contains('Home|Library or Community Center|Place of Worship').any(),
        dayData['condition'].str.contains('NoStress').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 1

    if true_conditions >= threshold:
        return 'Loneliness', true_conditions

    return None, 0


# Panicked emotion when a data row matches its constraints
def emotion_panicked(dayData):
    conditions = [
        dayData['HR'].mean() > 100 and dayData['HR'].mean() < 120,
        dayData['Skin_Temperature'].mean() > 33 or dayData['Skin_Temperature'].mean() < 36,
        dayData['Location'].str.contains(
            'Work|Public Transportation|Outside Activity|Unfamiliar Environment|High-pressure Environment').any(),
        dayData['condition'].str.contains('Interruption').any(),
    ]

    true_conditions = sum(conditions)
    threshold = 3

    if true_conditions >= threshold:
        return 'Panicked', true_conditions

    return None, 0


# Runs the current file and on the training data 
if __name__ == "__main__":
    file_Path = "Train.csv"
    found_emotions = detect_emotion(file_Path)
    print("Detected emotions:", found_emotions)
