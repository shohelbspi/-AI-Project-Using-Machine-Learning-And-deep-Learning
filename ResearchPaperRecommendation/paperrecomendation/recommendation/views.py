from django.shortcuts import render
from sentence_transformers import util
import torch

import pickle
import os

# Create your views here.
embeddings_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'word_embaddings.pkl')
sentences_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'sentences.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'model.pkl')

with open(embeddings_path, 'rb') as f:
    embeddings = pickle.load(f)

with open(sentences_path, 'rb') as f:
    sentences = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def remmendation(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        query_embedding = model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
        
        if cosine_scores.numel() > 0 and 5 <= cosine_scores.size(0):
            top_results = torch.topk(cosine_scores, k=5)
            
            recommended_sentences = []
            for idx in top_results[1].tolist():
                if 0 <= idx < len(sentences):
                    recommended_sentences.append(sentences[idx])
        else:
            recommended_sentences = []
        
        return render(request, 'index.html', {'recommended_sentences': recommended_sentences})
    
    return render(request, 'index.html')


