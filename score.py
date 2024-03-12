# Importing necessary libraries
from typing import Tuple


"""
Scores a trained model on a text.
Args:
    text (str): The input text to be scored.
    model: The trained model.
    threshold (float): The threshold for binary classification.
Returns:
    tuple: A tuple containing the prediction (bool) and propensity score (float).
"""

def score(text: str, model_data, threshold: float) -> Tuple[bool, float]:

    model = model_data['model']
    vectorizer = model_data['vectorizer']
    
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    
    # Predict probability
    probabilities = model.predict_proba(text_vector)[0]
    
    
    propensity = probabilities[1]  # Assuming the positive class is at index 1
    prediction = bool(propensity > threshold)
        
    return prediction, propensity
