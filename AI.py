import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_user_input(dataset): # Importing the workout types from the dataset.
    
    valid_workouts = dataset['Workout_Type'].unique().tolist() #Workout_Type is one of the data in my selected dataset
    
    # Get and validate workout type
    while True:
        print("\nAvailable workout types:", ', '.join(valid_workouts))
        workout_type = input("Enter the type of workout: ").strip() #Input command for the user to type in the workout they are doing.
        if workout_type in valid_workouts:
            break
        print("Invalid workout type! Please choose from the available options.") # Incase the user types a wrong workout which is not in dataset then error message appears.
    
    
    while True:
        try:
            duration = float(input("Enter the session duration (in hours): ")) # Input command for the session duration of the user.
            if 0 < duration <= 24:  # The duration avaialble.
                break # I wanted the duration to be in minutes for the algorithm to be more precise but my dataset had values only in 'hours'
            print("Please enter a reasonable duration between 0 and 24 hours.") # Input command for the duration.
        except ValueError:
            print("Please enter a valid number for duration!") 
    
    return workout_type, duration

# Load Dataset
try:
    data_path = 'Gym.csv.csv' # The data path to my dataset
    dataset = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Could not find the file {data_path}") # If the path or name is not matching then the error message aooears.
    exit(1)
except Exception as e:
    print(f"Error loading the dataset: {str(e)}")
    exit(1)

# This code is to create the column High_Calories
dataset['High_Calories'] = (dataset['Calories_Burned'] >= dataset['Calories_Burned'].median()).astype(int)

# X is the main targets for the algorithm to take the data from.
X = dataset[['Workout_Type', 'Session_Duration (hours)']]
y = dataset['High_Calories'] # y is for predicting the calories related to the data provided from the user.


encoder = OneHotEncoder(drop='first') # This function is for onehot encoding only for 'High_Calories'
X_encoded = encoder.fit_transform(X[['Workout_Type']]).toarray()

# Combine encoded categorical features with the numerical feature.
X_numerical = X[['Session_Duration (hours)']].values
X = np.hstack((X_encoded, X_numerical))

# Scaling the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training both the algorithms for its accuracy
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3) #The second algorithm.
knn.fit(X_train, y_train)

# These set of codes is to display the accuracies of both the algorithms.
y_pred_log = log_reg.predict(X_test) #Logistic Regression.
y_pred_knn = knn.predict(X_test)# KNeighborsClassifier.
print(f'\nLogistic Regression Accuracy: {accuracy_score(y_test, y_pred_log) * 100:.2f}%')
print(f'KNN Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%')

# Main prediction loop.
while True:
    print("\n=== Workout Calories Prediction ===")
    workout_type, duration = get_user_input(dataset)
    
    # The input data preparation.
    new_data = pd.DataFrame({
        'Workout_Type': [workout_type],
        'Session_Duration (hours)': [duration]
    })
    
    # Transforming the data.
    new_data_encoded = encoder.transform(new_data[['Workout_Type']]).toarray()
    new_data_numerical = new_data[['Session_Duration (hours)']].values
    new_data_combined = np.hstack((new_data_encoded, new_data_numerical))
    new_data_scaled = scaler.transform(new_data_combined)
    
    # The predicitons are being made in this 2 line of code.
    log_reg_pred = log_reg.predict(new_data_scaled)[0]
    knn_pred = knn.predict(new_data_scaled)[0]
    
    # Display results
    print("\nPrediction Results:")
    print(f"Workout Type: {workout_type}")
    print(f"Duration: {duration} hours")
    print(f"\nLogistic Regression predicts: {'High' if log_reg_pred == 1 else 'Low'} calories burn")
    print(f"KNN predicts: {'High' if knn_pred == 1 else 'Low'} calories burn")
    
    # Before ending the whole thing the algorithm asks the user if it wants to make another prediction or no.
    another = input("\nWould you like to make another prediction? (yes/no): ").lower()
    if another != 'yes':
        print("Thank you for using the Workout Calories Predictor!")
        break