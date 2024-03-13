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
from keybert import KeyLLM, KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer
import re
from sentence_transformers import SentenceTransformer
from flair.embeddings import TransformerDocumentEmbeddings
# %%
import nltk
#nltk.download('stopwords')


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

def get_comma_separated_abstracts(papers):
    abstracts = [paper['abstract'] for paper in papers]
    #print(abstracts)
    return ', '.join(abstracts)


def obtain_keywords(papers):
    # Create your LLM
    openai.api_key = "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"

    prompt = """
    I have the following document:
    [DOCUMENT]

    I also have the ontology concepts as the following as a reference or similar domain keywords to extract:

    Based on the information above, extract the keywords that best describe the topic or terms of the text.
    Make sure to only extract keywords that appear in the text.
    Use the following format separated by commas:
    <keywords>
    """
    llm = OpenAI(model="gpt-3.5-turbo", prompt=prompt, chat=True)
    #llm = 'm3rg-iitd/matscibert'
    # Load it in KeyLLM
    kw_model = KeyLLM(llm)
    #sentence_model = SentenceTransformer('m3rg-iitd/matscibert')
    #kw_model = KeyBERT(model=sentence_model)

    # Extract keywords
    keywords = kw_model.extract_keywords(papers, check_vocab=True)

    return keywords