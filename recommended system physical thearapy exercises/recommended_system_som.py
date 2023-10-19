import os
import pandas as pd

def load_data_from_directories(root_directory):
    data = []  # To store data from all subjects and exercises
    subject_data = []  # To store data for each subject
    for subject_folder in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject_folder)
        if os.path.isdir(subject_path):
            for exercise_folder in os.listdir(subject_path):
                exercise_path = os.path.join(subject_path, exercise_folder)
                if os.path.isdir(exercise_path):
                    for unit_folder in os.listdir(exercise_path):
                        unit_path = os.path.join(exercise_path, unit_folder)
                        if os.path.isdir(unit_path):
                            for file_name in os.listdir(unit_path):
                                if file_name == 'template_session.txt':
                                    file_path = os.path.join(unit_path, file_name)
                                    df = pd.read_csv(file_path, sep=';', header=0)
                                    data.append(df)
                                    # Add subject and exercise columns to the DataFrame
                                    df['subject'] = subject_folder
                                    df['exercise_type'] = exercise_folder
                                    subject_data.append(df)

    # Combine data for all subjects and exercises into one DataFrame
    data_df = pd.concat(data, ignore_index=True)
    subject_df = pd.concat(subject_data, ignore_index=True)
    
    return data_df, subject_df

# Specify the root directory where your data is stored
root_directory = r'C:\Users\adina\Desktop\תקיית עבודות\machine learning udemy\physical thearapy exercises SOM'

# Load the data from directories
dataset, subject_data = load_data_from_directories(root_directory)
dataset['subject'] = subject_data['subject']
# save some space 
subject_data= None
root_directory=None

# encoding for the subjects and the exercises 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ('encoder_subject', OneHotEncoder(), [10]),
        ('encoder_exercise', OneHotEncoder(), [11])
    ],
    remainder='passthrough' 
)

data_transformed = ct.fit_transform(dataset)

# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_transformed  = sc.fit_transform(data_transformed)


# splitting the data to training and test
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data_transformed, test_size=0.1, random_state=42)

# applying the SOM model 
import numpy as np
from minisom import MiniSom
som = MiniSom(x=30, y=30, input_len = training_set.shape[1] , sigma=1.0 , learning_rate= 0.5)
som.random_weights_init(training_set)
som.train_random(data= training_set, num_iteration=10000)

# Calculate the quantization error
quantization_error = som.quantization_error(training_set)
print("Quantization Error:", quantization_error)

# import os
import pandas as pd

def load_data_from_directories(root_directory):
    data = []  # To store data from all subjects and exercises
    subject_data = []  # To store data for each subject
    for subject_folder in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject_folder)
        if os.path.isdir(subject_path):
            for exercise_folder in os.listdir(subject_path):
                exercise_path = os.path.join(subject_path, exercise_folder)
                if os.path.isdir(exercise_path):
                    for unit_folder in os.listdir(exercise_path):
                        unit_path = os.path.join(exercise_path, unit_folder)
                        if os.path.isdir(unit_path):
                            for file_name in os.listdir(unit_path):
                                if file_name == 'template_session.txt':
                                    file_path = os.path.join(unit_path, file_name)
                                    df = pd.read_csv(file_path, sep=';', header=0)
                                    data.append(df)
                                    # Add subject and exercise columns to the DataFrame
                                    df['subject'] = subject_folder
                                    df['exercise_type'] = exercise_folder
                                    subject_data.append(df)

    # Combine data for all subjects and exercises into one DataFrame
    data_df = pd.concat(data, ignore_index=True)
    subject_df = pd.concat(subject_data, ignore_index=True)
    
    return data_df, subject_df

# Specify the root directory where your data is stored
root_directory = r'C:\Users\adina\Desktop\תקיית עבודות\machine learning udemy\physical thearapy exercises SOM'

# Load the data from directories
dataset, subject_data = load_data_from_directories(root_directory)
dataset['subject'] = subject_data['subject']
# save some space 
subject_data= None
root_directory=None

# encoding for the subjects and the exercises 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers=[
        ('encoder_subject', OneHotEncoder(), [10]),
        ('encoder_exercise', OneHotEncoder(), [11])
    ],
    remainder='passthrough' 
)

data_transformed = ct.fit_transform(dataset)

# feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
data_transformed  = sc.fit_transform(data_transformed)


# splitting the data to training and test
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data_transformed, test_size=0.1, random_state=42)

# applying the SOM model 
import numpy as np
from minisom import MiniSom
som = MiniSom(x=30, y=30, input_len = training_set.shape[1] , sigma=1.0 , learning_rate= 0.5)
som.random_weights_init(training_set)
som.train_random(data= training_set, num_iteration=10000)

# Calculate the quantization error
quantization_error = som.quantization_error(training_set)
print("Quantization Error:", quantization_error)

# Building the Recommendation System
bmu_indices = []
for x in test_set:
    bmu  = som.winner(x)
    bmu_indices.append(bmu)


bmu = None
# Create empty array to store recommended items
recommended_items = []
for bmu in bmu_indices:
    # Find the corresponding data point in the original dataset
    corresponding_data_point = data_transformed[bmu[1]]
    recommended_items.append(corresponding_data_point)

# inverse transforming the data from the recommended_items list 
encoder_subject = ct.named_transformers_['encoder_subject']
encoder_exercise = ct.named_transformers_['encoder_exercise']
recommended_items_inversed=[]

recommended_items= sc.inverse_transform(recommended_items[13:])

for item in recommended_items:
    subject = encoder_subject.inverse_transform([item[:5]])
    exercise = encoder_exercise.inverse_transform([item[5:13]])
    detailes = item[13:]
    item= {'Subject' : subject[0][0] , 'Exercise' : exercise[0][0], 'Detailes' : detailes}
    recommended_items_inversed.append(item)    
    
