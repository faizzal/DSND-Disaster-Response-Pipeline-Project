import nltk 

nltk.download(['punkt', 'wordnet','stopwords'])

import sys
import re
import numpy as np
import pandas as pd
import sqlite3
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split 
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names =  df.columns[4:]
    return X , Y , category_names

def tokenize(text):
    
    #1- apply normalize step on text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    #2- apply tokenize by Split text into words using NLTK
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    #apply sterm and lemmatizer
    lemmatizer = WordNetLemmatizer()
    new_tokens = []
    for i in tokens:
        tok = lemmatizer.lemmatize(i).strip()
        new_tokens.append(tok)
        
    return new_tokens


def build_model(): 
    pipeline = Pipeline([ 
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0))))
    ])
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)) 
        }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
        y_pred = model.predict(X_test)
        for i in range(Y_test.shape[1]):
            print("Classification Report Of" + Y_test.columns[i]+'\n'
                  ,'\n'+ classification_report(Y_test.values[:,i],y_pred[:,i])
                  , 'Accuracy:', accuracy_score(Y_test.values[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
        pickle.dump(model, open(model_filepath, 'wb')) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()