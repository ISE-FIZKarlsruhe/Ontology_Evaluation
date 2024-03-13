from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os.path
from gensim import corpora
from gensim.models import tfidfmodel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import nltk
from gensim.models import Phrases
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import os
import pickle
import torch
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer
import re
import os.path
from gensim import corpora
from gensim.models import tfidfmodel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import nltk
from gensim.models import Phrases
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rdflib import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import os
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer

def parse_titles_and_abstracts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # Using UTF-8 encoding
        text = file.read()

    # Splitting the text into chunks based on the pattern "Title:"
    chunks = re.split(r'\n?Title:', text)

    papers = []
    for chunk in chunks:
        if chunk.strip():  # Skip any empty chunks
            # Splitting each chunk into title and abstract
            title, abstract = chunk.split('Abstract:', 1)
            papers.append({'title': title.strip(), 'abstract': abstract.strip()})

    return papers

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model and tokenizer, and send the model to GPU
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device)

def encode_texts(texts):
    """Encode a list of texts into vectors using GPU."""
    with torch.no_grad():
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

query = ['Modulated structures']

file_path = '/home/mirza/ontology_evaluation/output.txt'  # Replace with your file path
papers = parse_titles_and_abstracts(file_path)
print(len(papers))
papers_less = papers[:100]
abstracts_list = [paper['abstract'] for paper in papers]

# Convert texts and query to vectors
text_vectors = encode_texts(papers_less)
query_vector = encode_texts(query)

# Convert texts to vectors
text_vectors = encode_texts(text_vectors)

# Move vectors to GPU for FAISS
res = faiss.StandardGpuResources()  # Use standard GPU resources
dimension = text_vectors.shape[1]
gpu_index = faiss.IndexFlatL2(dimension)  # L2 distance
gpu_index = faiss.index_cpu_to_gpu(res, 0, gpu_index)  # Move index to GPU
gpu_index.add(text_vectors)  # Add vectors to the index

# Encode and search a query
query = "Your query text here"
query_vector = encode_texts(query)  # Query converted to vector
k = 5  # Number of nearest neighbors to find
distances, indices = gpu_index.search(query_vector, k)

# Print the top k results with the full text
print("Top K similar texts:")
for i, idx in enumerate(indices[0]):
    print(f"{i + 1}: Text index {idx} with distance {distances[0][i]}")
    if idx < len(abstracts_list):  # Check if the index is within the range of 'texts' list
        print("Full Text:", abstracts_list[idx])
    else:
        print("Index out of range. Check if 'texts' list matches the indexed data.")
    print("-" * 50)

obtain_keywords(abstracts_list)
