
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Preprocess function
def preprocess_data(input_data):
    # Example preprocessing steps (adjust for your actual preprocessing logic)
    input_data['cab_type'] = input_data['cab_type'].map({'Uber': 0, 'Lyft': 1})
    input_data['destination'] = input_data['destination'].map({
        "Back Bay": 0, "Beacon Hill": 1, "Boston University": 2, "Fenway": 3,
        "Financial District": 4, "Haymarket Square": 5, "North End": 6, 
        "North Station": 7, "Northeastern University": 8, "South Station": 9,
        "Theatre District": 10, "West End": 11
    })
    input_data['weekday'] = input_data['weekday'].map({
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
        "Saturday": 5, "Sunday": 6
    })

    return input_data

# Load your trained XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model('xgboost_model.json')  # Ensure you have a saved model in this format
