#!/usr/bin/env python3.10
# coding: utf-8

import os
import sys
import warnings
import logging
import pandas as pd

# Suppress TensorFlow logs (except ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'")

# Set logging level for TensorFlow to ERROR after setting environment variable
logging.basicConfig(level=logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Additional logging setup for diagnostics
logging.basicConfig(filename='syntacticdepth.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Import SpaCy for NLP processing
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

print("Libraries and NLP model imported successfully.")

def clean_text(text):
    # Perform basic text cleaning and normalization
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return cleaned_text

def calculate_depth(token):
    if not list(token.children):
        return 1
    else:
        return 1 + max(calculate_depth(child) for child in token.children)

def analyze_constituency(text):
    cleaned_text = clean_text(text)
    logging.info(f"Processing cleaned text: {cleaned_text}")
    try:
        doc = nlp(cleaned_text)
        roots = 0
        total_depth = 0

        for sentence in doc.sents:
            root = [token for token in sentence if token.head == token][0]
            depth = calculate_depth(root)
            total_depth += depth
            roots += 1

        average_depth = total_depth / roots if roots > 0 else 0
        return average_depth
    except AssertionError:
        logging.error(f"AssertionError: Mismatch in sentence and tree counts for text: {cleaned_text}")
        return "nan"
    except Exception as e:
        logging.error(f"Exception {e} occurred for text: {cleaned_text}")
        return "nan"

# Read the CSV file
df = pd.read_csv("Official_AivsHUman.csv", encoding="UTF-8")
print("DataFrame loaded successfully.")

# Calculate syntactic depth for each text and store in DataFrame
df['Syntactic_Depth'] = df['Text'].apply(analyze_constituency)
print("Syntactic analysis completed.")

# Save the updated DataFrame to a new CSV file
df.to_csv("Official_AivsHUman.csv", index=False)
print("CSV file created successfully with updated syntactic depth.")

