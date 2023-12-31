﻿# SOM-applications
I made 2 SOM models for 2 appliactions :
1) SMS Spam Collection v.1 : The SMS Spam Collection v.1 is a dataset of SMS messages tagged as either "ham" (legitimate) or "spam." This dataset is valuable for SMS spam research and contains a total of 5,574 SMS messages in English. The messages have been collected from various sources, including web forums and academic theses. The dataset provides a balanced representation of ham and spam messages , and the target was to get a message and "cluster" her to spam or not, reduce dimentionality and find commom features between the messages.

2) Physical Therapy Exercise Recommendation System : The Physical Therapy Exercise Recommendation System is a machine learning-based solution designed to assist individuals in selecting appropriate physical therapy exercises. It utilizes a dataset containing exercise records from various subjects and exercise types.

Key features of the recommendation system include:

Data Collection: The system collects exercise data from multiple subjects, each performing various types of exercises. The data is organized into a structured format, including exercise details, subject information, and exercise types.

Data Preprocessing: Preprocessing steps are applied to the collected data, including one-hot encoding for subjects and exercise types, as well as feature scaling to ensure uniformity.

Self-Organizing Map (SOM): The system employs a Self-Organizing Map, an unsupervised learning algorithm, to group similar exercises based on patterns and characteristics. This clustering helps in making exercise recommendations.

Quantization Error: The quantization error, a measure of how well the SOM represents the data, is calculated. A lower quantization error indicates that the SOM captures exercise patterns effectively.

Exercise Recommendations: To make exercise recommendations, the system identifies exercises that are similar to those a user has performed or is interested in. It leverages the clustering created by the SOM to suggest exercises that align with the user's needs and goals.  
