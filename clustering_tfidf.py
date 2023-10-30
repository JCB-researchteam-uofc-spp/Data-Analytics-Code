###################################################
##### Date: February 04,2021
##### Revision: V1.6
##### File : cluster_tfidf_v1.6.py
##### Property of University of Calgary, Canada
##### Version History: 
###################################################

###################################################
###### Import Statements
###################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import collections
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import string


df = pd.read_csv('XXXXX.csv', encoding='utf-8')


## cleaning data code

def remove_tracks(text):
    text = str.lower(text)
    text = re.sub('(rt @[a-z0-9]+)\w+','', text)
    text = re.sub('(@[a-z0-9]+)\w+','', text)
    text = re.sub('(http\S+)', '', text)
    text = re.sub('([^0-9a-z \t])','', text)
    return text
df['clean_text'] = df['full_text'].apply(lambda x: remove_tracks(x))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
df['clean_text'] = df['clean_text'].apply(lambda x: remove_punct(x))

def tokenization(text):
    text = re.split('\W+', text)
    return text
df['clean_text'] = df['clean_text'].apply(lambda x: tokenization(x.lower()))


stop_words = nltk.corpus.stopwords.words('english')
sw_list = ['the','can','say','says','im','youre', 'hey', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'alberta', 'british columbia', 'manitoba', 'new brunswick', 'newfoundland and labrador', 'northwest territories', 'nova scotia', 'nunavut', 'ontario', 'prince edward island', 'saskatchewan', 'yukon', 'toronto', 'montreal', 'calgary', 'ottawa', 'edmonton', 'mississauga', 'north York', 'winnipeg', 'scarborough', 'vancouver', 'quebec', 'qc', 'hamilton', 'brampton', 'surrey', 'laval', 'halifax', 'etobicoke', 'london', 'okanagan', 'victoria', 'Newand', 'andi', 'amp', 'ottcity', 'theyre', 'vax', 'vaxx']
stop_words.extend(sw_list)

def remove_stopwords(text):
    text = [word for word in text if word not in stop_words]
    return text
    
df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords(x))

stop_words_french = nltk.corpus.stopwords.words('french')

def remove_stopwords_french(text):
    text = [word for word in text if word not in stop_words_french]
    return text
    
df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords_french(x))

text = df['clean_text']
data = []
for item in (text):
    item = ' '.join(item)
    item = item.lstrip()
    data.append(item)
cleaned_df = text.to_frame()
tweets_arr = np.array(data)
embeddings = model.encode(tweets_arr)
print("shape of embeddings",embeddings.shape, file=open("shape of embeddings_us_cluster.txt", "w"))
# import pickle
# output = open('embeddings_cluster_anti.pkl', 'wb')
# for i in range(0, len(embeddings)):
#     pickle.dump(embeddings[i], output)
# output.close()

tweets = cleaned_df
stop_words = set(stopwords.words('english'))
stop_words.update(('You', 'tha', 'in', 'ha', 'says', 'user', 'get', 'heres', 'they', 'can', 'th', 'hear', 'told', 'im', 'didn', 'didnt', 'he', 'she', 'would', 'here', 'dont', 'go', 'even', 'it', 'don', 'tha', 'is', 'say', 'you', 'in', 'theyve', 'they', 'th', 'can', 'happen', 'coz', 'isn', 'everyth', 'they', 'mine', 'meeee', 'one', 'http', 'said'))
def get_top_words(documents, top_n):
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_df=0.3)
    tfidf_matrix = vectoriser.fit_transform(documents)
    feature_names = vectoriser.get_feature_names()
    df_tfidf = pd.DataFrame()
    for doc in range(len(documents)):
        words = []
        scores = []
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            words.append(w)
            scores.append(s)
        df_temp = pd.DataFrame(data={'word':words, 'score':scores})
        df_temp = df_temp.sort_values('score',ascending=False).head(top_n)
        df_temp['topic'] = doc
        df_tfidf = df_tfidf.append(df_temp)
    return df_tfidf

# Clustering and topic modelling

cluster = AgglomerativeClustering(n_clusters=20, linkage='ward')  
predictions = cluster.fit_predict(embeddings)
pred_arr = np.array(predictions)

df1 = df
df2 = pd.DataFrame(predictions)
df = pd.concat([df1, df2], axis=1)
df.to_csv('XXX_cluster.csv', index=True, header = True)

topic_docs = []
for topic in range(17):
    l = tweets.loc[pred_arr==topic]['clean_text'].values
    #s = (' '.join([t for t in l if str(t) != 'nan']))
    s = (','.join([str(t) for t in l if str(t) != 'nan']))
    s = ' '.join([word for word in s.split() if word not in stop_words])
    topic_docs.append(s)
df_tfidf = get_top_words(topic_docs, 10)
# Put limit for score
df_tfidf = df_tfidf[(df_tfidf.score > 0.08)]
pd.set_option('display.max_rows', df_tfidf.shape[0]+1)
df_tfidf.to_csv('XXX.csv', index = None, header = True)
topics = df_tfidf['topic'].value_counts()
