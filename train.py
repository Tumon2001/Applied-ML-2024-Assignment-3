import re
import csv
import nltk
import joblib
import sklearn
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords

nltk.download("stopwords")

# Loading raw data (email spam)
raw_data = pd.read_csv("raw_data.csv")

# Preprocessing data
def preprocess_data(data):

    # Remove characters other than English letters and digits
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    # Convert to lowercase
    data['text'] = data['text'].apply(lambda x: x.lower())

    # Remove stopwords
    s = set(stopwords.words("english"))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in s and word]))

    return data

data = preprocess_data(raw_data)

# Splitting the preprocessed data
# Splits: 80% training data, 5% validation data, 15% test data
def split_data(data, test_size = 0.2, validation_size = 0.25, output_path = './'):

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size = test_size, random_state = 1)

    # Further split the test data into validation and test sets
    validation_data, test_data = train_test_split(test_data, test_size = validation_size, random_state = 1)

    train_data.to_csv(f'{output_path}/train.csv', index = False)
    validation_data.to_csv(f'{output_path}/validation.csv', index = False)
    test_data.to_csv(f'{output_path}/test.csv', index = False)

split_data(data)

# Loading preprocessed train data
X_train = pd.read_csv("train.csv")
y_train = X_train['spam']
X_train_text = X_train['text']

# Loading preprocessed validation data
X_validation = pd.read_csv("validation.csv") 
y_validation = X_validation['spam']
X_validation_text = X_validation['text']

# Loading preprocessed test data
X_test = pd.read_csv("test.csv")
y_test = X_test['spam']
X_test_text = X_test['text']

# Data vectorization
tfidf_vectorizer = TfidfVectorizer(max_features = 50000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_validation_tfidf = tfidf_vectorizer.transform(X_validation_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Training Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Saving the trained model and vectorizer to a pickle file
joblib.dump({'model': model, 'vectorizer': tfidf_vectorizer}, 'trained_logistic_regression.pkl')
