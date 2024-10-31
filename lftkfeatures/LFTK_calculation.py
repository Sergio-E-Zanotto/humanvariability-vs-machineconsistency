#!/usr/bin/env python3.10
# coding: utf-8

import os
import sys
import warnings

# Suppress TensorFlow logs (except ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'")

# Set logging level for TensorFlow to ERROR after setting environment variable
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import other libraries after configuring environment and warning suppression
import pandas as pd
import numpy as np
import tensorflow as tf  # Import TensorFlow after setting the environment variable
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Import spaCy and lftk last after suppressing warnings
import spacy
import lftk

print("Libraries imported successfully.")
print("Libraries imported successfully.")

# Path to the uploaded JSONL file
file_path = 'subtaskB_train.jsonl'

# Initialize lists to hold the data
texts = []
models = []
sources = []
labels = []
ids = []

# Read the JSONL file
print("Reading JSONL file...")
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        texts.append(data['text'])
        models.append(data['model'])
        sources.append(data['source'])
        labels.append(data['label'])
        ids.append(data['id'])
        
print("JSONL file read successfully.")

# Create a DataFrame
df = pd.DataFrame({
    'Text': texts,
    'Model': models,
    'Source': sources,
    'Label': labels,
    'ID': ids
})

print("DataFrame created successfully.")



nlp = spacy.load("en_core_web_sm")
texts = df["Text"]

print("spaCy model loaded successfully.")

docs = [nlp(text) for text in texts]

# Initialize the LFTK Extractor with your docs
LFTK = lftk.Extractor(docs=docs)

# Optional: Customize the feature extraction
LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)

print("Feature extraction customized successfully.")

# List of feature families to process
feature_families = [
    "wordsent", "worddiff", "partofspeech", "entity",
    "avgwordsent", "avgworddiff", "avgpartofspeech", "avgentity",
    "lexicalvariation", "typetokenratio", "readformula", "readtimeformula"
]

# Initialize the final DataFrame for features
final_df = pd.DataFrame()

# Initialize a separate DataFrame for storing mean values of feature families
mean_df = pd.DataFrame()

print("Processing feature families...")
k = 0
# Process each feature family
for family in feature_families:
    k += 1
    print(k)
    # Search for features in the current family
    features_current = lftk.search_features(family=family, return_format="list_key")
    
    # Extract features for the current family, resulting in a list of dictionaries
    extracted_features_list = LFTK.extract(features=features_current)
    
    # Convert the list of dictionaries (each dict per document) into a DataFrame
    features_df = pd.DataFrame(extracted_features_list)
    
    # Calculate the mean of each row (document) across the current family's features,
    # ignoring non-numeric columns if present
    family_mean = features_df.mean(axis=1)
    
    # Add this family's mean to the mean_df with a specific column name
    mean_df[family + '_mean'] = family_mean
    
    # If it's the first family, initialize final_df with these features
    if final_df.empty:
        final_df = features_df
    else:
        # Merge the new features into the final DataFrame
        final_df = pd.concat([final_df, features_df], axis=1)

# Handling potential duplicate columns if the same feature is present in multiple families in final_df
final_df = final_df.loc[:,~final_df.columns.duplicated()]

print("Feature families processed successfully.")

Official_df = pd.concat([df, final_df], axis=1)

print("DataFrames concatenated successfully.")

Official_df.to_csv("Official_AivsHuman.csv", header=True, index=False)

print("CSV file created successfully.")
