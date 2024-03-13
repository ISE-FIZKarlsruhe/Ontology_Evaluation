from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
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
from transformers import AutoModelForMaskedLM, AutoTokenizer
import re
import os.path
from gensim.models import tfidfmodel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import nltk
from gensim.models import Phrases
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rdflib import Graph
import numpy as np
from sklearn.metrics import precision_score, recall_score
import os
import pickle
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
import re
#sys.path.append('../')
from evaluation.ontology import sparql, walk
from evaluation import ontology
from tqdm import tqdm
from term_extractor import *
import itertools
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch


def calculate_similarity(excel_path):
    # Load the tokenizer and model from the pre-trained BERT
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

    # Function to encode text into embeddings
    def encode(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # List to store similarity scores
    similarity_scores = []

    # Calculate embeddings and similarity for each row
    for _, row in df.iterrows():
        concept_embedding = encode(row['Concepts'])
        keyword_embedding = encode(row['Keywords'])
        similarity = cosine_similarity(concept_embedding.detach().numpy(), keyword_embedding.detach().numpy())[0][0]
        similarity_scores.append(similarity)

    # Add the similarity scores to the DataFrame
    df['Similarity Score'] = similarity_scores

    # Return the modified DataFrame
    return df

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


# Load a pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

def encode_texts(texts):
    """Encode a list of texts into vectors."""
    with torch.no_grad():
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        model_output = model(**encoded_input)
        # Use mean pooling for sentence-level embeddings
        embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# # Example texts and query
###read ontologies
# %%
results =f"""

            SELECT distinct ?id ?label ?comment WHERE {{
                
                OPTIONAL{{ ?id rdfs:label ?label . }}
                #OPTIONAL{{ ?id rdfs:comment ?comment . }}
            }}
        """

# %%
namespaces = """
    prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    prefix owl: <http://www.w3.org/2002/07/owl#>
    prefix skos: <http://www.w3.org/2004/02/skos/core#>
    prefix ns2: <http://www.w3.org/2006/time#>
"""
def separate_words(s):
    # Use regex to split camelCase or PascalCase strings
    words = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    return words.lower()

def _owl_get_classes(graph) -> list[tuple[str, str]]:
    results = graph.query(
        # f"""
        #     {namespaces}

        #     SELECT distinct ?id ?label ?comment ?skoslabel WHERE {{
        #         {{
        #             ?id a owl:Class .
                    
        #         }} UNION {{
        #             ?id a rdfs:Class .
        #         }} UNION {{
        #             ?subclass rdfs:subClassOf ?id .
        #         }} UNION {{
        #             ?id a owl:ObjectProperty .
        #         }} UNION {{
        #             ?id a owl:DatatypeProperty .
        #         }} UNION {{
        #             ?id a owl:NamedIndividual .
        #         }}
        #         OPTIONAL {{ ?id rdfs:label ?label . }}
        #         OPTIONAL {{ ?id rdfs:comment ?comment . }}
        #         OPTIONAL {{ ?id skos:prefLabel ?skoslabel . }}
        #         FILTER (!isBlank(?id))
        #     }}
        # """
        f"""
            {namespaces}
            SELECT distinct ?id ?label ?comment ?skoslabel WHERE {{
                {{
                    ?id a owl:Class .
                }} UNION {{
                    ?id a rdfs:Class .
                }} UNION {{
                    ?subclass rdfs:subClassOf ?id .
                }} UNION {{
                    ?id a owl:ObjectProperty .
                }} UNION {{
                    ?id a owl:DatatypeProperty .
                }} UNION {{
                    ?id a owl:NamedIndividual .
                }}
                OPTIONAL {{
                        ?id rdfs:label ?label .
                        FILTER(LANG(?label) = "en")
                    }}
                OPTIONAL {{
                        ?id rdfs:comment ?comment .
                        FILTER(LANG(?comment) = "en")
                    }}
                OPTIONAL {{
                        ?id skos:prefLabel ?skoslabel .
                        FILTER(LANG(?skoslabel) = "en")
                    }}
                FILTER (!isBlank(?id))
            }}
        """
    )
    id_labels = []
    for result in results:
        if result.id and not result.label and not result.skoslabel:
            id_labels.append(( result.id.toPython(), separate_words(result.id.toPython().split('#')[-1] if '#' in result.id.toPython() else result.id.toPython().split('/')[-1])))
        elif result.label:
            #id_labels.append((result.id.toPython(), result.label.value))
            id_labels.append((separate_words(result.id.toPython().split('#')[-1] if '#' in result.id.toPython() else result.id.toPython().split('/')[-1]), result.label.value.lower()))
        elif result.skoslabel:
            id_labels.append((separate_words(result.id.toPython().split('#')[-1] if '#' in result.id.toPython() else result.id.toPython().split('/')[-1]), result.skoslabel.value.lower()))
        elif result.comment:
            id_labels.append((separate_words(result.id.toPython().split('#')[-1] if '#' in result.id.toPython() else result.id.toPython().split('/')[-1]), result.comment.value.lower()))
    return id_labels

# %%

stop_counter = 0

ontology_name = []
ontology_concepts_list_final = []
abstracts_list_final = []
obtained_keywords = []

for file_name in os.listdir('./ontologies'):
    #print(file_name)
    ontology_name.append(file_name)
    if stop_counter == 20:
        break
    # Check if the file is a Turtle or OWL file
    if file_name.endswith('.ttl') or file_name.endswith('.owl'):
        file_path = os.path.join('./ontologies', file_name)
        #ontology_evaluation/ontologies/AMONTOLOGY.owl
        #file_path = './ontologies/DISO.ttl'    
        # Assuming ontology.sparql.graph_from is a method to read the ontology file
        # and 'ontology' is a module or object that needs to be imported or defined earlier
        g = ontology.sparql.graph_from(file_path)
        #g = ontology.sparql.graph_from('/home/mirza/ontology_evaluation/ontologies/DEB.owl')
        ontology_concepts_list = _owl_get_classes(g)

        ontology_concepts = set()
        for concept, i in ontology_concepts_list:
            ontology_concepts.add(i)

        ontology_concepts_list = list(ontology_concepts_list)
        #print(ontology_concepts_list)
        #exit()
        descriptive_strings = [description for _, description in ontology_concepts_list]
        ontology_concepts_list_final.append(descriptive_strings)
        #ontology_document = ' '.join([i.replace('_', ' ') for concept, i in ontology_concepts_list])
        #print(ontology_document)
        #print(type(ontology_document))
        #exit()
        #print('ontology concepts are ', list(ontology_concepts))
        ####
        query = ontology_concepts_list

        file_path = './output.txt'  # Replace with your file path
        papers = parse_titles_and_abstracts(file_path)
        #print(len(papers))
        papers_less = papers[:100]
        abstracts_list = [paper['abstract'] for paper in papers_less]
        #print(abstracts_list)
        # Convert texts and query to vectors
        text_vectors = encode_texts(abstracts_list)
        #exit()
        #print(query)
        #exit()
        query_vector = encode_texts(query)

        # Create and train the FAISS index
        dimension = text_vectors.shape[1]  # Dimension of the vectors
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
        index.add(text_vectors)  # Add vectors to the index

        # Perform the search
        k = 10  # Number of nearest neighbors to find
        distances, indices = index.search(query_vector, k)

        similar_abstracts = []

        print("Top K similar texts:")
        for i, idx in enumerate(indices[0]):
            print(f"{i + 1}: Text index {idx} with distance {distances[0][i]}")
            if idx < len(abstracts_list):  # Check if the index is within the range of 'texts' list
                #print("Full Text:", abstracts_list[idx])
                similar_abstracts.append(abstracts_list[idx])
            else:
                print("Index out of range. Check if 'texts' list matches the indexed data.")
            print("-" * 50)
        abstracts_list_final.append(similar_abstracts)
        keywords = obtain_keywords(similar_abstracts)
        #print('similar keywords for top 10 abstracts ', keywords)
        flattend_keywords = [s_keywords for s_keywords in keywords]
        flattened_list = [item for sublist in flattend_keywords for item in sublist] 
        #print('flattend keywords are ', flattened_list)
        #exit()
        obtained_keywords.append(flattened_list)
        print("#" * 50)
        #break
        stop_counter += 1

# ontology_concepts_list_final = [item for sublist in ontology_concepts_list_final for item in sublist]
# abstracts_list_final = [item for sublist in abstracts_list_final for item in sublist]
# obtained_keywords = [item for sublist in obtained_keywords for item in sublist]

#print('ontology names are ', ontology_name)
#print('ontology concepts are ', ontology_concepts_list_final)
#print('similar abstracts are ', abstracts_list_final)
#print('obtained keywords are ', obtained_keywords)

# Placeholder lists for demonstration
# ontology_name = ["Ontology 1", "Ontology 2", "Ontology 3"]
# ontology_concepts_list_final = [["Concept 1.1", "Concept 1.2"], ["Concept 2.1"], ["Concept 3.1", "Concept 3.2", "Concept 3.3"]]
# obtained_keywords = [["Keyword 1.1", "Keyword 1.2"], ["Keyword 2.1", "Keyword 2.2", "Keyword 2.3"], ["Keyword 3.1"]]

# Zipping the lists together to form rows
rows = zip(ontology_name, ontology_concepts_list_final, obtained_keywords)

# Creating a DataFrame from the zipped lists
df_three_columns = pd.DataFrame(rows, columns=['Ontology Name', 'Concepts', 'Keywords'])

# Convert lists in 'Concepts' and 'Keywords' columns to strings
df_three_columns['Concepts'] = df_three_columns['Concepts'].apply(lambda x: '; '.join(x))
df_three_columns['Keywords'] = df_three_columns['Keywords'].apply(lambda x: '; '.join(x))

# Saving to Excel
three_columns_excel_path = "./results/ontology_evaluation/ontology_keywords_concepts.xlsx"
df_three_columns.to_excel(three_columns_excel_path, index=False)

# Optionally, save the results to a new Excel file
df_results = calculate_similarity("./results/ontology_keywords_concepts.xlsx")
df_results.to_excel("./results/ontology_keywords_concepts_with_bert_score_20.xlsx", index=False)


