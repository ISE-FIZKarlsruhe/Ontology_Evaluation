# %%
import csv
import requests
import json

# Function to fetch title and abstract using DOI
def fetch_subject(doi):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        title = data['message'].get('title', ["N/A"])[0]
        abstract = data['message'].get('abstract', "N/A")
        subject = data['message'].get('subject', "N/A")
        if subject == "N/A":
            return []
        else:
            return subject
    else:
        print(f"Failed to fetch data for DOI {doi}. Status code: {response.status_code}")
        return "N/A", "N/A"



# %%
fetch_subject('10.1016/j.jnoncrysol.2013.01.048')

# %%
import requests
import csv

def get_elsevier_data(doi, api_key):
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            title = data['full-text-retrieval-response']['coredata']['dc:title']
            abstract = data['full-text-retrieval-response']['coredata']['dc:description']
            #subject = [item['$'] for item in data['full-text-retrieval-response']['coredata']['dcterms:subject']]
            subject = fetch_subject(doi)
        except KeyError:
            print(f"Title or abstract not found for DOI {doi}.")
            return "N/A", "N/A", "N/A"
        return title, abstract, subject
    else:
        print(f"Failed to retrieve data for DOI {doi}. HTTP Status Code: {response.status_code}")
        return "N/A", "N/A", "N/A"

# Replace 'your_api_key_here' with your actual Elsevier API key
api_key = "eacb970be4b0bbf5eb5311e7f1832bf2"



# %%
get_elsevier_data('10.1016/j.jnoncrysol.2013.01.048', api_key)

# %%
# Read DOIs from CSV
dois = []
with open('dois.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) > 1:
            dois.append(row[1].strip())

# %%
# Write titles and abstracts to a text file
with open('output_1.txt', 'w', encoding='utf-8') as txtfile:
    for doi in dois[:10]:
        title, abstract, subject = get_elsevier_data(doi, api_key)
        try:
            txtfile.write(f"Title:{title.strip()} Abstract:{abstract.strip()} Subject:{subject} \n\n")
        except:
            print(doi)
        #print(f"Fetched data for DOI: {doi}")


