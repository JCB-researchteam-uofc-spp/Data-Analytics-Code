# -*- coding: utf-8 -*-
"""Building Social Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f33gUbSTuJwAi_ecZcYEVSNis97Aju2L
"""

import sys
import json
import re
import numpy as np
from datetime import datetime
import pandas as pd

tweetfile = 'ukraine_CDN_26jan-10may2022.csv'

retweets = pd.read_csv(tweetfile)

retweets.columns

# 1. Export edges from Retweets
retweets['original_twitter'] = retweets['text'].str.extract('RT @([a-zA-Z0-9]\w{0,}):', expand=True)

edges = retweets[['name', 'original_twitter']]
edges.columns = ['Source', 'Target']

edges2 = edges.groupby(['Source','Target']).count()
edges2 = edges2.reset_index()

# Export nodes from the edges and add node attributes for both Sources and Targets.
users = retweets[['name','followers']]
users = users.sort_values(['name','followers'], ascending=[True, False])
users = users.drop_duplicates(['name'], keep='first')

ids = edges2['Source'].append(edges2['Target']).to_frame()
ids['Label'] = ids
ids.columns = ['name', 'Label']
ids = ids.drop_duplicates(['name'], keep='first')
nodes = pd.merge(ids, users, on='name', how='left')

print(nodes.shape)
print(edges2.shape)

edges2.columns = ['target','source']

# Export nodes and edges to csv files
edges2.to_csv('ukraine_CDN_26jan-10may2022_edge.csv', encoding='utf-8', index=False)

