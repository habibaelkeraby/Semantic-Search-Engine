# Imports required
import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import string
import gensim
import operator
import re
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
import spacy_streamlit
import warnings

warnings.filterwarnings('ignore')

scrapping_df = pd.read_json('companyScrapped.json')

spacy_nlp = spacy.load('en_core_web_sm')
#spacy_nlp =spacy.load('lib_spacy/spacy-models-en_core_web_sm-3.1.0/spacy-models-en_core_web_sm-3.1.0/meta/en_core_web_sm-3.0.0a1')
#spacy_nlp = spacy.load('en_core_web_sm-3.0.0a1.json')
#spacy_nlp = en_core_web_sm_abd.load()
#spacy_nlp = "en_core_web_sm"

#create list of punctuations and stopwords
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

#function for data cleaning and processing
#This can be further enhanced by adding / removing reg-exps as desired.

def spacy_tokenizer(sentence):

    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)

    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)

    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)

    #creating token object
    tokens = spacy_nlp(sentence)

    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]

    #return tokens
    return tokens

scrapping_df['webSiteText_tokenized'] = scrapping_df['webSiteText'].map(lambda x: spacy_tokenizer(x))

company_text = scrapping_df['webSiteText_tokenized']

series = pd.Series(np.concatenate(company_text)).value_counts()[:100]
wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)

plt.figure(figsize=(15,15), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#creating term dictionary
dictionary = corpora.Dictionary(company_text)

#filter out terms which occurs in less than 4 documents and more than 20% of the documents.
#NOTE: Since we have smaller dataset, we will keep this commented for now.

#dictionary.filter_extremes(no_below=4, no_above=0.2)

#list of few which which can be further removed
stoplist = set('hello and if this can would should could tell ask stop come go')
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

#print top 50 items from the dictionary with their unique token-id
dict_tokens = [[[dictionary[key], dictionary.token2id[dictionary[key]]] for key, value in dictionary.items() if key <= 50]]

corpus = [dictionary.doc2bow(desc) for desc in company_text]

word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

company_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)
company_lsi_model = gensim.models.LsiModel(company_tfidf_model[corpus], id2word=dictionary, num_topics=300)

gensim.corpora.MmCorpus.serialize('company_tfidf_model_mm', company_tfidf_model[corpus])
gensim.corpora.MmCorpus.serialize('company_lsi_model_mm', company_lsi_model[company_tfidf_model[corpus]])

#Load the indexed corpus
company_tfidf_corpus = gensim.corpora.MmCorpus('company_tfidf_model_mm')
company_lsi_corpus = gensim.corpora.MmCorpus('company_lsi_model_mm')

company_index = MatrixSimilarity(company_lsi_corpus, num_features = company_lsi_corpus.num_terms)

def search_similar_companies(search_term):

  query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
  query_tfidf = company_tfidf_model[query_bow]
  query_lsi = company_lsi_model[query_tfidf]

  company_index.num_best = 5

  company_list = company_index[query_lsi]

  company_list.sort(key=itemgetter(1), reverse=True)
  company_names = []

  for j, company in enumerate(company_list):

    company_names.append (
        {
            'Relevance': round((company[1] * 100),2),
            'Company Details': scrapping_df['company'][company[0]],
            'Company Name': scrapping_df["company"][company[0]].get('name'),
            'Website Text': scrapping_df['webSiteText'][company[0]],
            'Website Links': scrapping_df['links'][company[0]],
        }

    )
    if j == (company_index.num_best-1):
        break

  return pd.DataFrame(company_names, columns=['Relevance','Company Details','Company Name','Website Text','Website Links'])

# search for companies that are related to below search parameters
# st.write(search_similar_companies('halal'))
#---- Streamlit App Interface ----#
# Customize layout
st.set_page_config(layout="wide")

st.title("Semantic Search Engline")
st.subheader("Output search results ranked by relevance based on semantic similarity with keyword input")

#Expandable section for general instructions
with st.expander("Follow the steps below to search the database"):
  st.write("""1. Use the search bar below to input your search keywords.
            \n2. Once you click search, the results will be displayed, sorted by highest relevance.
            """)
# User input using form and single line text input
with st.form("my_form"):
   search_keyword = st.text_input("Search Bar", "I am searching for...",key="placeholder")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Search")
   if submitted:
       st.write(search_similar_companies(search_keyword))
