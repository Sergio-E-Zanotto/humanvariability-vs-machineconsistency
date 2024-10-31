#!/usr/bin/env python3.10
# coding: utf-8

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("AivsHUman_sem.csv", encoding="UTF-8")
print("DataFrame loaded successfully.")

# Load the NRC Emotion Intensity Lexicon
lexicon = pd.read_csv("NRC-Emotion-Intensity-Lexicon-v1.txt", sep='\t', header=None)
lexicon.columns = ['Word', 'Emotion', 'Intensity']

# Create a dictionary for quick lookup
lexicon_dict = {}
for index, row in lexicon.iterrows():
    if row['Word'] not in lexicon_dict:
        lexicon_dict[row['Word']] = {}
    lexicon_dict[row['Word']][row['Emotion']] = row['Intensity']

# Function to preprocess and tokenize text
def preprocess_and_tokenize(text):
    text = text.lower().strip()
    tokens = text.split()
    return tokens

# Function to calculate emotion scores
def calculate_emotion_scores(text, lexicon_dict):
    tokens = preprocess_and_tokenize(text)
    emotion_scores = {}
    
    for token in tokens:
        if token in lexicon_dict:
            for emotion, intensity in lexicon_dict[token].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(float(intensity))
    
    # Calculate average intensity for each emotion
    for emotion in emotion_scores:
        emotion_scores[emotion] = np.mean(emotion_scores[emotion])
    
    return emotion_scores

# Calculate emotion scores for each text in the dataset
emotion_data = []
for text in df['Text']:
    scores = calculate_emotion_scores(text, lexicon_dict)
    emotion_data.append(scores)

# Create a DataFrame from the emotion scores
emotion_df = pd.DataFrame(emotion_data).fillna(0)

# Concatenate the emotion scores with the original dataset
result_df = pd.concat([df, emotion_df], axis=1)

# Print the updated DataFrame
print("Updated DataFrame with Emotion Scores:")
print(result_df.head())

# Save the updated DataFrame to a new CSV file
result_df.to_csv("AivsHUman_with_emotion_scores.csv", index=False)
print("CSV file created successfully with emotion scores.")
