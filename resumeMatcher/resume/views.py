from django.shortcuts import render
import PyPDF2
import docx
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.


def home(request):
    if request.method == 'POST':
        job_description = request.POST.get('job_description', '').strip()
        resume_files = request.FILES.getlist('resumes')

        resumes = []
        filenames = []

        for resume_file in resume_files:
            text = extract_text(resume_file)
            if text.strip():
                resumes.append(text)
                filenames.append(resume_file.name)

        if not job_description or not resumes:
            return render(request, 'resume/home.html', {
                'message': "Please upload resumes and enter a job description."
            })

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Cosine similarity
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 matches
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        return render(request, 'resume/home.html', {
            'message': "Top matching resumes:",
            'top_resumes': zip(top_resumes, similarity_scores)
        })

    return render(request, 'resume/home.html')



def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    else:
        return ""


