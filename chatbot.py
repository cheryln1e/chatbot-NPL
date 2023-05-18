#!/usr/bin/env python
# coding: utf-8


import nltk
import random
import numpy as np
import math
import csv
import re
from math import log10
from scipy import spatial
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.linear_model import LogisticRegression


#tokenization
def tokenize(doc):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tok_doc = []
    tok_doc = tokenizer.tokenize(doc)
    return tok_doc


#compute term log frequency
def logfreq_weighting(vector):
    lf_vector = []
    for frequency in vector:
        lf_vector.append(log10(1+frequency))
    return np.array(lf_vector)


#get similarity
def get_similarity(vector, vector_dic):    
    similarity={}
    for q in vector_dic:
        sim = 1-spatial.distance.cosine(vector, vector_dic[q])
        similarity[q]=sim
    return sorted(similarity.items(), key=lambda x: x[1], reverse=True) 


#intent matching
def intent_clf(query):
#load data set
    intent_dir = {
        'identity':'identity.txt',
        'small_talk':'small_talk.txt',
        'QA':'QA.txt'
    }
    
    data=[]
    intents=[]
    for intent in intent_dir.keys():
        path=intent_dir[intent]
        with open(path, encoding='utf8', mode='r') as f:
            for line in f:
                data.append(line)
                intents.append(intent)
# term-document matrix                
    stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    def stemmed_words(doc): 
        return (stemmer.stem(w) for w in analyzer(doc))
    stem_vectorizer = CountVectorizer(analyzer=stemmed_words)
    train_counts = stem_vectorizer.fit_transform(data) 
# tf-idf
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(train_counts) 
    train_tf = tfidf_transformer.transform(train_counts) 
# classifier
    clf = LogisticRegression(random_state=0).fit(train_tf, intents)
    processed_query=stem_vectorizer.transform(query)
    processed_query=tfidf_transformer.transform(processed_query)
    return clf.predict(processed_query)


def small_talk(stem_query):
    Q=[]
    A=[]
    Q_dic={}
    A_dic={}

# load dataset and store in dictionary
    with open ('small_talk_dataset.csv','r') as f:
        csv_reader=csv.reader(f)
        for row in csv_reader:
            Q.append(row[0])
            A.append(row[1])
    Q1=list(set(Q))
    n=1
    for i in Q1:
        Q_dic[n]=i
        A_dic[n]=[]
        index=Q.index(i)
        for j in Q:  
            if j == i:
                A_dic[n].append(A[index]) 
                index+=1
        n+=1

# tokenization  
    tok_Q={}
    for q in Q_dic:        
        tok_Q[q]=tokenize(Q_dic[q])

# lower case
    filted_Q={}
    for q in tok_Q:
        filted_Q[q]=[w.lower() for w in tok_Q[q]]
# stemming
    stemmer = SnowballStemmer('english')
    stem_Q = {}
    for q in filted_Q:
        stem_Q[q]=[stemmer.stem(w) for w in filted_Q[q]]  

# get volcabulary   
    vocal=[]
    for q in stem_Q:
        for w in stem_Q[q]:
            if w not in vocal:
                vocal.append(w) 
# bag-of-word and vectors               
    bow={}
    for q in stem_Q:
        bow[q]=np.zeros(len(vocal))
        for w in stem_Q[q]:
            index=vocal.index(w)
            bow[q][index]+=1         
# log weighting
    for q in bow:
        bow[q]=logfreq_weighting(bow[q])

        
# map query to vector
    vector_query=np.zeros(len(vocal))
    for stem in stem_query:
        try:
            index = vocal.index(stem)
            vector_query[index] += 1
        except ValueError:
            continue
    vector_query = logfreq_weighting(vector_query)

# compute similarity and sort
    similarity = get_similarity(vector_query, bow) 
    index=similarity[0][0]
    print(random.choice(A_dic[index]))


def Q_A(stem_query):
    Q=[]
    A=[]
    Q_dic={}
    A_dic={}
    with open ('QA_dataset.csv','r') as f:
        csv_reader=csv.reader(f)
        for row in csv_reader:
            Q.append(row[0])
            A.append(row[1])
    Q1=list(set(Q))
    n=1
    for i in Q1:
        Q_dic[n]=i
        A_dic[n]=[]
        index=Q.index(i)
        for j in Q:  
            if j == i:
                A_dic[n].append(A[index]) 
                index+=1
        n+=1
# tokenization    
    tok_Q={}
    for q in Q_dic:        
        tok_Q[q]=tokenize(Q_dic[q])
#remove stopwords
    filted_Q={}
    stop_words=stopwords.words('english')
    for q in tok_Q:
        filted_Q[q]=[w.lower() for w in tok_Q[q] if w.lower() not in stop_words]
# stemming
    stemmer = SnowballStemmer('english')
    stem_Q = {}
    for q in filted_Q:
        stem_Q[q]=[stemmer.stem(w) for w in filted_Q[q]]  
    
# Create vocabulary
    vocal=[]
    for q in stem_Q:
        for w in stem_Q[q]:
            if w not in vocal:
                vocal.append(w) 
    
# Create bag-of-words and vectors           
    bow={}
    for q in stem_Q:
        bow[q]=np.zeros(len(vocal))
        for w in stem_Q[q]:
            index=vocal.index(w)
            bow[q][index]+=1         
    for q in bow:
        bow[q]=logfreq_weighting(bow[q])

#normalise query
    stem_query = [w for w in stem_query if w not in stop_words] 
# map query to vector
    vector_query=np.zeros(len(vocal))
    for stem in stem_query:
        try:
            index = vocal.index(stem)
            vector_query[index] += 1
        except ValueError:
            continue
    vector_query = logfreq_weighting(vector_query)

#get similarity
    similarity = get_similarity(vector_query, bow)
    index=similarity[0][0]
    print(random.choice(A_dic[index]))





def identity(query):
    global user
    q_name=['what is my name', 
       'what do you call me', 
       'who am i',
      'do you know who i am',
      'tell me my name']
    f_query=''.join([c.lower() for c in query])    
# pattern matching
    x=re.findall("(?:name is|call me|I am|I'm) (\w+)", query)
    if x!=[]:
        user=x[0]
        output=[]
        output.append("Hi {}, what can I do for you?".format(user))
        output.append("Hello {}, how can I help you?".format(user))
        print(random.choice(output))
    elif f_query in q_name:
        if user=='':
            print("Please tell me your name:")
        else:
            print(f"You are {user}!")
    else:
        print("Sorry, I don't understand")



# main

stop = False
user=''
query = input("Hello! I'm your chatbot, What would you like me to call you? (enter 'bye' for quit)\n")
while not stop:    
    if query == 'bye':
        stop = True
        print('see you next time!')
        break

#intent matching
    intent=intent_clf([query])
# Tokenization
    tok_query = tokenize(query)    
#normalise query
    stemmer = SnowballStemmer('english')
    stem_query = [stemmer.stem(word.lower()) for word in tok_query]    

    if intent == 'identity':       
        identity(query)
    elif intent == 'small_talk':
        small_talk(stem_query)
    elif intent == 'QA':
        Q_A(stem_query)
    
    query=input()

