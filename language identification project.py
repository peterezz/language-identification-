# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:35:33 2021

@author: Peter
"""

import pandas as pd
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import speech_recognition as sr
print("Hello")
print("<<Eng-Wael>> ")
print("We Wish  you a nice day")
print("PLease Select your an option: ")

print("Enter  1  if you Choose to identify a text Query")
print("Enter  2  if you Choose to input identify a voice Query")

opt = int(input("Enter Your number"))
if (opt!=1 and opt!=2):
    
    print("Program exit due to Wrong intput")
else:
    stopwords = nltk.corpus.stopwords.words('english')  
    ps = nltk.PorterStemmer() 
# Cleaning up text 

    def clean_text(text):    
    
        #stopword removal
        
        toke=list(text.split(' '))
       
    
        #punctutaion removal
        t_lator=str.maketrans('','',string.punctuation)
        text=text.translate(t_lator)
        remove_digits = str.maketrans('', '', string.digits)
        text = text.translate(remove_digits)    
    
        #removing special symbol
        for i in '“”—':
            text = text.replace(i, ' ')
        
        return text

    def clean_data(df):
        df.dropna(how='any')

    vectorizer = TfidfVectorizer()
    def train_data(df):
 
        X = vectorizer.fit_transform(df['text'])
        true_k = 22
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
                print
        return model
    
    print("-------------------------------------------------------")
    print("All languages in The Program ")
    print("-------------------------------------------------------")
    df = pd.read_csv("dataset2.csv")
    df.dropna(how='any') #clean the dataset
    df.columns=['text','language']
    df = df.sort_values(['language'])
    i = 0
    while i < 22000:
         print(df.iat[i, 1])
         i += 1000
    print("-------------------------------------------------------")
    print("learning from the dataset.... ")
    print("-------------------------------------------------------")
    clean_data(df)
    train=train_data(df)
    if(opt == 2 ):
        print("-------------------------------------------------------")
        print("voice input ")
        print("-------------------------------------------------------")
        
        r = sr.Recognizer()
        with sr.Microphone() as source:
                print("speak")
                audio= r.record(source, duration=5)
        try:
            print("recognising...")
            text= r.recognize_google(audio)
            print('{}'.format(text))
        except:
            print("try again!")
        clean_text(text)
        
    if(opt == 1 ):
        print("-------------------------------------------------------")
        print("text input ")
        print("-------------------------------------------------------")
        
        text = input("Please enter a string:\n")
        clean_text(text)
    def predicti (model):
        Y = vectorizer.transform([text])
        prediction = model.predict(Y)
        print (prediction)
    
        z= prediction[0]*1000
        print("the language is: ",df.iat[z, 1])

         
    print("-------------------------------------------------------")
    print("model prediction")
    print("-------------------------------------------------------")
    predicti(train)
    
       
        