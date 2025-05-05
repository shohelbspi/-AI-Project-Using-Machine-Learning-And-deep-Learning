from django.shortcuts import render
import nltk
import re
from nltk import PorterStemmer
import os
import pickle
import numpy as np

logistic_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'logistic_regresion.pkl')
tfidf_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'tfidf.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'label_encoder.pkl')

with open(logistic_path, 'rb') as logistic_file:
    logistic = pickle.load(logistic_file)

with open(tfidf_path, 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

with open(encoder_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Create your views here.


nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


def predicttion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf.transform([cleaned_text])

    predict = logistic.predict(input_vectorized)
    predict = encoder.inverse_transform(predict)
    label =  np.max(logistic.predict(input_vectorized))

    return predict,label

def emotion_view(request):
    predicted_emotion = None
    probability = None

    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            predicted_emotion, probability = predicttion(user_input)

    return render(request, 'emotion_form.html', {
        'predicted_emotion': predicted_emotion[0] if predicted_emotion else None,
        'probability': probability
    })
