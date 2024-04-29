# %% [markdown]
# **Source**: https://www.youtube.com/watch?v=xF2UJTmRU_Y

# %%


# %%
import csv
import os
from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from keybert.llm import TextGeneration
from keybert import KeyLLM, KeyBERT
from sentence_transformers import SentenceTransformer

# %%
from huggingface_hub import login
login("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")

# %%
import bibtexparser
from yake import KeywordExtractor
from rake_nltk import Rake
from sklearn.metrics import precision_recall_fscore_support
from yake import KeywordExtractor
from rake_nltk import Rake
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from pyate import combo_basic, basic, cvalues
from summa import keywords as summa_keywords
import spacy
import pandas as pd
from keybert import KeyBERT
from nltk.stem import PorterStemmer
from Levenshtein import distance
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords

# %%
import nltk
nltk.download('stopwords')

# %%
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model_mistral = CAutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50,
    hf=True
)

# Tokenizer
tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Pipeline
generator_mistral = pipeline(
    model=model_mistral, tokenizer=tokenizer_mistral,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)

# %%
spacy.load("en_core_web_lg")
spacy.load("en_core_web_sm")

# %%
import torch

# %%
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_mixtral = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer_mixtral = AutoTokenizer.from_pretrained(model_id)

generator_mixtral = pipeline(
    "text-generation",
    model=model_mixtral, tokenizer=tokenizer_mixtral, max_new_tokens=100,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)


# %%
example_prompt = """
<s>[INST]
I have the following document:
- Localized magnetic hyperthermia using magnetic nanoparticles (MNPs) under the application of small magnetic fields is a promising tool for treating small or deep-seated tumors.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] localized magnetic hyperthermia,magnetic nanoparticles (MNPs),magnetic fields</s>"""

keyword_prompt = """
[INST]

I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

prompt = example_prompt + keyword_prompt

# Mistral7B
llm_mistral = TextGeneration(generator_mistral, prompt=prompt)
llm_mixtral = TextGeneration(generator_mixtral, prompt=prompt)

#kw_model = KeyLLM(llm)
#Mistral7B_keywords = kw_model.extract_keywords([abstract])[0]

# Mistral7B_embeddings
#model = SentenceTransformer('BAAI/bge-small-en-v1.5')
#embeddings = model.encode([abstract], convert_to_tensor=True)
#Mistral7B_embeddings_keywords = kw_model.extract_keywords([abstract], embeddings=embeddings, threshold=.5)[0]

# Mistral7B_KeyBERT
kw_model_mistral = KeyBERT(llm=llm_mistral, model='BAAI/bge-small-en-v1.5')
kw_model_mixtral = KeyBERT(llm=llm_mixtral, model='BAAI/bge-small-en-v1.5')

# %%
def extract_keywords_from_abstract(abstract):

    # Mistral7B
    Mistral7B_KeyBERT_keywords = kw_model_mistral.extract_keywords([abstract], threshold=.5)[0]
    Mixtral7B_KeyBERT_keywords = kw_model_mixtral.extract_keywords([abstract], threshold=.5)[0]

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))
    abstract = ' '.join([word for word in abstract.split() if word.lower() not in stop_words])

    # Initialize Spacy, YAKE, and RAKE keyword extractors
    nlp = spacy.load("en_core_web_lg")
    kw_extractor = KeywordExtractor() # KeywordExtractor(lan="en", n=3, dedupLim=0.6, dedupFunc='seqm', windowsSize=1, top=20, features=None)
    rake_nltk_var = Rake()

    # Extract keywords using Spacy entities
    doc = nlp(abstract)
    spacy_entities = [ent.text for ent in doc.ents]

    # Extract keywords using Spacy noun chunks
    doc = nlp(abstract)
    spacy_noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Extract keywords using YAKE
    yake_keywords = kw_extractor.extract_keywords(abstract)
    yake_keywords = [keyword[0] for keyword in yake_keywords]

    # Extract keywords using RAKE
    rake_nltk_var.extract_keywords_from_text(abstract)
    rake_keywords = rake_nltk_var.get_ranked_phrases()

    # Extract keywords using Pyate
    pyate_combo_basic_keywords = combo_basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_basic_keywords = basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_cvalues_keywords = cvalues(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()

    # Extract keywords using summa
    summa_keywords_ = [keyword[0] for keyword in summa_keywords.keywords(abstract, scores=True)]
    
    # Extract keywords using KeyBERT
    keybert_model = KeyBERT()#KeyBERT(model="m3rg-iitd/matscibert")#KeyBERT()
    keybert_keywords = [keyword[0] for keyword in keybert_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 3), stop_words='english')] #keyphrase_ngram_range=(1, 3),
    
    # Extract keywords using KeyBERT+MatSciBERT
    keybert_m_model = KeyBERT(model="m3rg-iitd/matscibert")#KeyBERT()
    keybert_m_keywords = [keyword[0] for keyword in keybert_m_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 3), stop_words='english')] #keyphrase_ngram_range=(1, 3),

    # Extract keywords using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform([abstract])
    tfidf_keywords = tfidf_vectorizer.get_feature_names_out()

    # Extract keywords using LSA
    lsa_model = TruncatedSVD(n_components=10)  # Adjust the number of components as needed
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
    lsa_keywords = [tfidf_keywords[i] for i in lsa_model.components_[0].argsort()[::-1]]

    # Extract keywords using LDA
    dictionary = Dictionary([abstract.split()])
    corpus = [dictionary.doc2bow(abstract.split())]
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in [abstract.split()]]
    lda_model = LdaModel(corpus=doc_term_matrix, num_topics=10, id2word=dictionary)  # Adjust the number of topics as needed
    lda_keywords = [word for word, _ in lda_model.show_topic(0)]

    return {
        "Spacy_entities": spacy_entities,
        "Spacy_noun_chunks": spacy_noun_chunks,
        "YAKE_keywords": yake_keywords,
        "RAKE_keywords": rake_keywords,
        "Pyate_combo_basic_keywords": pyate_combo_basic_keywords,
        "Pyate_basic_keywords": pyate_basic_keywords,
        "Pyate_cvalues_keywords": pyate_cvalues_keywords,
        "Summa_keywords": summa_keywords_,
        "Keybert_keywords": keybert_keywords,
        "Keybert_m_keywords": keybert_m_keywords,
        "TFIDF_keywords": tfidf_keywords,
        "LSA_keywords": lsa_keywords,
        "LDA_keywords": lda_keywords,
        #"Mistral7B": Mistral7B_keywords,
        #"Mistral7B_embeddings": Mistral7B_embeddings_keywords,
        "Mistral7B_KeyBERT": Mistral7B_KeyBERT_keywords,
        "Mixtral7B_KeyBERT": Mixtral7B_KeyBERT_keywords,
    }


# %%
# Example usage
abstract = "Functionalization facilitates targeted delivery of these nanoparticles to various cell types, bioimaging, gene delivery, drug delivery and other therapeutic and diagnostic applications."
keywords = extract_keywords_from_abstract(abstract)
for method, extracted_keywords in keywords.items():
    print(method + ": ", extracted_keywords)

# %%
# Function to tokenize and stem text
def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    if isinstance(text, str):
        tokens = [stemmer.stem(word) for word in text.split()]
        return ' '.join(tokens)
    else:
        return str(text)

# Function to calculate Levenshtein distance similarity
def levenshtein_similarity(text1, text2):
    return 1 - (distance(text1, text2) / max(len(text1), len(text2)))

# Function to find synonyms using WordNet
def find_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Function to calculate cosine similarity using TF-IDF
def cosine_similarity_score(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix)[0][1]

# Function to calculate fuzzy matching score
def fuzzy_matching_score(text1, text2):
    return fuzz.token_set_ratio(text1, text2)

# Function to evaluate keywords
def evaluate_keywords(ground_truth_keywords, extracted_keywords):
    ground_truth_keywords = list(set(ground_truth_keywords))
    extracted_keywords = list(set(extracted_keywords))
    # Initialize variables for evaluation metrics
    tp, fp, fn = 0, 0, 0

    # Tokenize and stem ground truth keywords
    ground_truth_stems = [tokenize_and_stem(keyword) for keyword in ground_truth_keywords]

    # Iterate over extracted keywords
    for extracted_keyword in extracted_keywords:
        # Tokenize and stem extracted keyword
        extracted_stem = tokenize_and_stem(extracted_keyword)

        # Check if extracted keyword matches any ground truth keyword
        matched = False
        for ground_truth_stem in ground_truth_stems:
            # Calculate similarity scores
            #levenshtein_sim = levenshtein_similarity(extracted_stem, ground_truth_stem)
            #cosine_sim = cosine_similarity_score(extracted_stem, ground_truth_stem)
            #fuzzy_score = fuzzy_matching_score(extracted_keyword, ground_truth_stem)

            # If any similarity score exceeds threshold, consider it a match
            #if levenshtein_sim > 0.8 or cosine_sim > 0.8 or fuzzy_score > 80:
            if extracted_stem == ground_truth_stem:
                matched = True
                break

        # Update evaluation metrics based on match status
        if matched:
            tp += 1
        else:
            fp += 1

    # Calculate false negatives (missed ground truth keywords)
    fn = len(ground_truth_keywords) - tp

    # Calculate precision, recall, and F1-score
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

# %%
def evaluate_keywords_from_bib(bib_file, extraction_functions, output_folder):
    # Load the BibTeX file
    with open(bib_file, 'r', encoding='utf-8') as bibfile:
        bib_database = bibtexparser.load(bibfile)

    # Initialize dictionaries to store cumulative scores
    cumulative_precision = {method: 0 for method in extraction_functions}
    cumulative_recall = {method: 0 for method in extraction_functions}
    cumulative_f1_score = {method: 0 for method in extraction_functions}
    total_abstracts = 0

    # Initialize lists to store ground truth keywords, extracted keywords, and evaluation results
    all_extracted_keywords = []
    all_evaluation_results = []
    all_evaluation_results_avg = []

    # Iterate over entries in the BibTeX file
    for entry in bib_database.entries:
        # Check if the entry has abstract and keywords
        if 'abstract' in entry and 'keywords' in entry:
            abstract = entry['abstract'].lower()
            ground_truth_keywords = entry['keywords'].split(',')
            total_abstracts += 1

            # Evaluate keywords for each extraction function
            for method, extraction_function in extraction_functions.items():
                extracted_keywords = extraction_function(abstract)
                precision, recall, f1_score = evaluate_keywords(ground_truth_keywords, extracted_keywords)

                # Accumulate scores
                cumulative_precision[method] += precision
                cumulative_recall[method] += recall
                cumulative_f1_score[method] += f1_score

                # Append data for CSV output
                all_evaluation_results.append((ground_truth_keywords, extracted_keywords, method, precision, recall, f1_score))

    # Calculate averages
    average_precision = {method: cumulative_precision[method] / total_abstracts for method in extraction_functions}
    average_recall = {method: cumulative_recall[method] / total_abstracts for method in extraction_functions}
    average_f1_score = {method: cumulative_f1_score[method] / total_abstracts for method in extraction_functions}

    # Print average scores
    print("Average Scores over all Abstracts:")
    for method in extraction_functions:
        print(f"Method      , Average Precision:                    , Average Recall:                    , Average F1-score:                    ")
        print(f"{method},{average_precision[method]},{average_recall[method]},{average_f1_score[method]}")
        all_evaluation_results_avg.append((method, average_precision[method], average_recall[method], average_f1_score[method]))

    # Write ground truth keywords, extracted keywords, and evaluation results to CSV files
    with open(os.path.join(output_folder, 'evaluation_results.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Method', 'Precision', 'Recall', 'F1-score'])
        writer.writerows(all_evaluation_results)
    
    with open(os.path.join(output_folder, 'evaluation_results_avg.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Precision', 'Recall', 'F1-score'])
        writer.writerows(all_evaluation_results_avg)

# Define extraction functions
extraction_functions = {
    "Spacy_entities": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_entities"],
    "Spacy_noun_chunks": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_noun_chunks"],
    "YAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["YAKE_keywords"],
    "RAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["RAKE_keywords"],
    "Pyate_combo_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_combo_basic_keywords"],
    "Pyate_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_basic_keywords"],
    "Pyate_cvalues_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_cvalues_keywords"],
    "Summa_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Summa_keywords"],
    "Keybert_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Keybert_keywords"],
    "Keybert_m_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Keybert_m_keywords"],
    "TFIDF_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["TFIDF_keywords"],
    "LSA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LSA_keywords"],
    "LDA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LDA_keywords"],
    #"Mistral7B": lambda abstract: extract_keywords_from_abstract(abstract)["Mistral7B"],
    #"Mistral7B_embeddings": lambda abstract: extract_keywords_from_abstract(abstract)["Mistral7B_embeddings"],
    "Mistral7B_KeyBERT": lambda abstract: extract_keywords_from_abstract(abstract)["Mistral7B_KeyBERT"],
    "Mixtral7B_KeyBERT": lambda abstract: extract_keywords_from_abstract(abstract)["Mixtral7B_KeyBERT"],
}

# %%
# Example usage
bib_file = "nanomaterials-v01-i01_20240418_texts/nanomaterials-v01-i01_20240418.bib"

# Specify the output folder
output_folder = "Results_llm/"

evaluate_keywords_from_bib(bib_file, extraction_functions, output_folder)

# %%


# %%



