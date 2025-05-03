from django.shortcuts import render
import re
import pickle
import PyPDF2
import docx
from scanning.forms import ResumeUploadForm


# Create your views here.


import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
svc_model = pickle.load(open(model_path, 'rb'))
tfidf = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models', 'tfidf.pkl'), 'rb'))
encoder = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.pkl'), 'rb'))

def cleanResume(text):
    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'@\w+|#\w+', ' ', text)

    text = re.sub(r'\brt\b|\bcc\b', ' ', text)

    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    escaped_punctuation = re.escape(punctuation)
    text = re.sub(f"[{escaped_punctuation}]", ' ', text)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_text(file, extension):
    if extension == 'pdf':
        reader = PyPDF2.PdfReader(file)
        return ''.join([page.extract_text() for page in reader.pages])
    elif extension == 'docx':
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif extension == 'txt':
        try:
            return file.read().decode('utf-8')
        except:
            return file.read().decode('latin-1')
    return ""

def predict_resume_category(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['resume']
            ext = file.name.split('.')[-1].lower()
            text = extract_text(file, ext)
            cleaned = cleanResume(text)
            vectorized = tfidf.transform([cleaned])
            pred_label = svc_model.predict(vectorized.toarray())
            category = encoder.inverse_transform(pred_label)[0]
            return render(request, 'upload.html', {'category': category})
    else:
        form = ResumeUploadForm()
    return render(request, 'upload.html', {'form': form})