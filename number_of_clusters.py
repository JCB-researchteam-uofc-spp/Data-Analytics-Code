###################################################
##### Date: February 20,2021
##### File : Number_of_clusters.py
##### Property of University of Calgary, Canada
###################################################

###################################################
###### Import Statements
###################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.cluster import AgglomerativeClustering
import collections
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
import re
import nltk
import pickle
from nltk.corpus import stopwords
import sys
import string
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import pairwise_distances
matplotlib.use('Agg')
plt.ioff()

df = pd.read_csv('XXX.csv', encoding='utf-8')

## cleaning data code

def remove_tracks(text):
    text = str.lower(text)
    text = re.sub('(rt @[a-z0-9]+)\w+','', text)
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
    #text = ''.join(text)
    #text = text.replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')   
    return text
df['clean_text'] = df['clean_text'].apply(lambda x: tokenization(x.lower()))


stop_words = nltk.corpus.stopwords.words('english')
sw_list = ['the','can','say','says','im','youre', 'hey', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'alberta', 'british columbia', 'manitoba', 'new brunswick', 'newfoundland and labrador', 'northwest territories', 'nova scotia', 'nunavut', 'ontario', 'prince edward island', 'saskatchewan', 'yukon', 'toronto', 'montreal', 'calgary', 'ottawa', 'edmonton', 'mississauga', 'north York', 'winnipeg', 'scarborough', 'vancouver', 'quebec', 'qc', 'hamilton', 'brampton', 'surrey', 'laval', 'halifax', 'etobicoke', 'london', 'okanagan', 'victoria']
stop_words.extend(sw_list)

def remove_stopwords(text):
    text = [word for word in text if word not in stop_words]
    return text
    
df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords(x))
df.head(5)

text = df['clean_text']
data = []
for item in (text):
    item = ' '.join(item)
    item = item.lstrip()
    data.append(item)
tweets_arr = np.array(data)
embeddings = model.encode(tweets_arr)


scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(embeddings)

rangeofcluster = range(5,30)

km_scores= []
km_silhouette = []
db_score = []

for i in rangeofcluster:
    km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
    preds = km.predict(X_scaled)
    
    print("Score for number of cluster(s) {}: {}".format(i,km.score(X_scaled)))
    km_scores.append(-km.score(X_scaled))
    
    silhouette = silhouette_score(X_scaled,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(X_scaled,preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))


plt.title("The elbow method for determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in rangeofcluster],y=km_scores,s=150,edgecolor='k')
plt.grid(True)
matplotlib.use('Agg')
plt.ioff()
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("K-means score",fontsize=15)
plt.xticks([i for i in rangeofcluster],fontsize=14)
plt.yticks(fontsize=15)
plt.savefig('1_Elbow_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png', dpi = 100)
#plt.show()

plt.figure(figsize=(7,4))
matplotlib.use('Agg')
plt.ioff()
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in rangeofcluster],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in rangeofcluster],fontsize=14)
plt.yticks(fontsize=15)
plt.savefig('3_silhouette_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png', dpi = 100)
#plt.show()

matplotlib.use('Agg')
plt.ioff()
plt.title("The Davies Bouldin method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in rangeofcluster],y=db_score,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Davies Bouldin score",fontsize=15)
plt.xticks([i for i in rangeofcluster],fontsize=14)
plt.yticks(fontsize=15)
plt.savefig('5_Davies_Bouldin_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png', dpi = 100)
#plt.show()


# Yellowbrick
# silhouette
model = KMeans()
visualizer = KElbowVisualizer(model, k=(rangeofcluster), metric='silhouette', timings=False, locate_elbow=True)
visualizer.fit(X_scaled) 
matplotlib.use('Agg')
plt.ioff()
visualizer.poof(outpath="4_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png")

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(rangeofcluster), timings=False)
visualizer.fit(X_scaled)        # Fit the data to the visualizer
matplotlib.use('Agg')
plt.ioff()
visualizer.poof(outpath="2_Elbow_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png")

# calinski_harabasz
model = KMeans()
visualizer = KElbowVisualizer(model, k=(rangeofcluster), metric='calinski_harabasz', timings=False)
visualizer.fit(X_scaled) 
matplotlib.use('Agg')
plt.ioff()
visualizer.poof(outpath="6_calinski_harabasz_ukraine_CDN_26jan-10may2022_pro_Russia_RUS.png")
