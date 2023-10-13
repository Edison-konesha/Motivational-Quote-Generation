import pandas as pd
import random
from datetime import datetime, timedelta


# mocked the heart rate because the heart rate was very low and i wanted to
# have a mixture of emotions to predict
def increase_heart_rate(df, percentage):
    num_rows = int(df.shape[0] * percentage)
    indices_to_modify = random.sample(range(df.shape[0]), num_rows)

    for idx in indices_to_modify:
        df.loc[idx, 'HR'] = random.randint(100, 200)

    return df


# generate locations so that i could use them on them when writing the functions
# add locations based on some constraints set for all the rows of the data
def generate_locations(df, max_gym_per_day, min_gym_hours_diff, min_time_at_location):
    random_locations = []
    daily_gym_count = 0
    last_gym_index = -1
    last_location_change_index = 0
    in_sleep_period = False

    locations = [
        'Home',
        'Work',
        'Grocery Store',
        'Gym',
        'Park or Outdoor Recreation Area',
        'Restaurant or Cafe',
        "Friend's or Family Member's House",
        'Shopping Mall or Retail Store',
        'Medical or Dental Office',
        'Gas Station or Convenience Store',
        'Place of Worship',
        'Library or Community Center',
        'Public Transportation',
        'Movie Theater or Entertainment Venue',
        'Outside activity'
    ]
    non_gym_locations = ['Home',
                         'Work',
                         'Grocery Store',
                         'Park or Outdoor Recreation Area',
                         'Restaurant or Café',
                         "Friend's or Family Member's House",
                         'Shopping Mall or Retail Store',
                         'Medical or Dental Office',
                         'Gas Station or Convenience Store',
                         'Place of Worship',
                         'Library or Community Center',
                         'Public Transportation',
                         'Movie Theater or Entertainment Venue',
                         'Outside activity',
                         'sleep'
                         ]

    for i in range(df.shape[0]):
        current_hour = df.loc[i, 'DateTime'].hour

        if i == 0:
            location = random.choice(locations)
            random_locations.append(location)
            continue

        time_since_last_change = (
                df.loc[i, 'DateTime'] - df.loc[last_location_change_index, 'DateTime']).total_seconds()

        if in_sleep_period:
            location = 'Sleep'
            sleep_duration = time_since_last_change / 3600

            if 7 <= sleep_duration < 8 and random.random() < 0.1:
                in_sleep_period = False
                last_location_change_index = i
        elif time_since_last_change < min_time_at_location * 60:
            location = random_locations[-1]
        else:
            if 22 <= current_hour <= 23 and not in_sleep_period and random.random() < 0.1:
                location = 'Sleep'
                in_sleep_period = True
            else:
                while True:
                    if daily_gym_count < max_gym_per_day:
                        location = random.choice(locations)
                    else:
                        location = random.choice(non_gym_locations)

                    if location != random_locations[-1]:
                        break

            last_location_change_index = i

        if location == 'Gym':
            if last_gym_index != -1 and (df.loc[i, 'DateTime'] - df.loc[
                last_gym_index, 'DateTime']).total_seconds() < min_gym_hours_diff * 3600:
                location = random.choice(non_gym_locations)
            else:
                daily_gym_count += 1
                last_gym_index = i

        if (i + 1) % (
                5 * 12 * 24) == 0:  # Reset the daily gym count after each day
            daily_gym_count = 0

        random_locations.append(location)

    return random_locations


# Load the data for the training and testing
data = 'Test.csv'
df = pd.read_csv(data)
# Start date and time
start_date_time = datetime(2023, 2, 1)  # start date and time
time_interval = timedelta(seconds=12)  # so the interval is 12 seconds, 5 data points per minute,

# Generate date and time for each heart rate
date_times = [start_date_time + i * time_interval for i in range(df.shape[0])]
# Add the date and time column
df['DateTime'] = date_times

# Increase heart rate for 40% of the rows
percentage_to_increase = 0.4  # 40% of rows will have heart rate above 100
df = increase_heart_rate(df, percentage_to_increase)

max_gym_per_day = 2  # maximum number of times a person can be in the gym
min_gym_hours_diff = 10  # difference between each time going at the gym
min_time_at_location = 40  # minimum time staying at the location
random_locations = generate_locations(df, max_gym_per_day, min_gym_hours_diff, min_time_at_location)
df['Location'] = random_locations

# Create lists of temperature ranges
temperature_ranges = [(5, 10), (10, 15), (15, 20), (20, 30)]
random_temperatures = [random.uniform(*random.choice(temperature_ranges)) for _ in range(df.shape[0])]
df['Weather'] = random_temperatures

skin_temperature_range = (35, 40)
random_skin_temperatures = [random.uniform(*skin_temperature_range) for _ in range(df.shape[0])]
# Add the skin temperature column to the data set
df['Skin_Temperature'] = random_skin_temperatures

# Save the modified dataset to a new file
df.to_csv('modifiedTest.csv', index=False)
