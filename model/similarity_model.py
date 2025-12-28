import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(user_skill, job_skills):
    """Return cosine similarity scores between the user_skill string and each string in job_skills.

    This function is defensive: if job_skills is empty it returns an empty numpy array,
    and if vectorization fails it returns a zero array of the correct length.
    """
    try:
        if not job_skills:
            return np.array([])

        docs = [str(user_skill)] + [str(s) for s in job_skills]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(docs)
        similarity = cosine_similarity(vectors[0:1], vectors[1:])
        return similarity[0]
    except Exception:
        return np.zeros(len(job_skills))
