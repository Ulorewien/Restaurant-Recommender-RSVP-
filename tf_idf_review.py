#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm


# In[2]:


nltk.download('stopwords')


# In[3]:


business_file = "final_business_CA.gzip"
reviews_file = "final_review_CA.gzip"
users_file = "final_data_user_yelp.gzip"


# In[4]:


### Do data preprocessing containing dictionary of user, businesses and star ratings
reviews_df = pd.read_pickle(reviews_file)


# In[5]:


reviews_df.sample(frac=1)


# In[6]:


users_review_df = reviews_df.groupby('user_id')[['text', 'date']].agg(list).reset_index()


# In[7]:


business_review_df = reviews_df.groupby('business_id')[['text', 'date']].agg(list).reset_index()


# In[21]:


def text_preprocess(review):
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    stop_W = stopwords.words("english")
    r = ''.join([c for c in review.lower() if (not c in punctuation) and c.isalpha()])
    word_list = []
    for w in r.split():
        w = stemmer.stem(w)
        if w not in stop_W:
            word_list.append(w)
    return word_list
    


# In[28]:


model = TfidfVectorizer(stop_words='english')


# In[29]:


tf_idf_df = {'user_id': [], 'business_id': [], 'tf_idf_hours_matrix': []}


# In[30]:


for idx, df in tqdm(reviews_df.iterrows()):
    user_id = df['user_id']
    business_id = df['business_id']
    user_corpus = users_review_df[users_review_df['user_id'] == user_id]['text'].tolist()[0]
    business_corpus = business_review_df[business_review_df['business_id'] == business_id]['text'].tolist()[0]
    tfidf = model.fit_transform(user_corpus + business_corpus)
    user_tidf = tfidf[:len(user_corpus)]
    business_tfidf = tfidf[len(user_corpus):]
    similarity_matrix = (user_tidf @ business_tfidf.T).toarray()
    user_review_hours = [(pd.Timestamp.now() - timestamp) / pd.Timedelta(hours=1) for timestamp in
                         users_review_df[users_review_df['user_id'] == user_id]['date'].tolist()[0]]
    business_review_hours = [(pd.Timestamp.now() - timestamp) / pd.Timedelta(hours=1) for timestamp in
                            business_review_df[business_review_df['business_id'] == business_id]['date'].tolist()[0]]
    hours_array = np.transpose([np.tile(user_review_hours, len(business_review_hours)),
                                np.repeat(business_review_hours, len(user_review_hours))]).reshape((len(user_review_hours), len(business_review_hours), 2))
    tfidf_feature = np.concatenate((similarity_matrix[..., np.newaxis], hours_array), axis=2)
    tf_idf_df["user_id"].append(user_id)
    tf_idf_df["business_id"].append(business_id)
    tf_idf_df["tf_idf_hours_matrix"].append(tfidf_feature)
    


# In[12]:


tf_idf_dataframe = pd.DataFrame(tf_idf_df)


# In[ ]:


tf_idf_dataframe.to_pickle("final_CA_tf_idf.gzip")


# In[ ]:




