from flask import Flask, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import realtime
import pickle
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn import naive_bayes
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import flask
from sklearn.externals import joblib
import os
from flask import render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import *
from nltk.probability import FreqDist
import nltk
import regex
from flask_ngrok import run_with_ngrok
from itertools import tee, islice
import tweepy
import time
import csv
import json
from tweepy import OAuthHandler
import re
import codecs
import sys
access_token = "103589925-PYiNRi6sAoSAFCau7Q5zDAqF7Kt8WwsK5EunWL3I"
access_token_secret = "u8N1nS93eN5npmtBOAxCwJgZE0W4wPCNe1CEuCB9lEIys"
consumer_key = "IIFBxSZv8YnhRJuvvDJVkR4ht"
consumer_secret = "zou3XOVMDXp9esingDNEowUeEPmTKkY4daZGYdmalovyd9JCxr"
def words_per_line(doc):
    minNgramLength=1
    maxNgramLength=1

    # analyze each line of the input string seperately
    for ln in doc.split('\n'):

        # tokenize the input string (customize the regex as desired)
        terms = regex.findall(u'(?u)\\b\\w+\\b', ln)

        # loop ngram creation for every number between min and max ngram length
        for ngramLength in range(minNgramLength, maxNgramLength+1):

            # find and return all ngrams
            # for ngram in zip(*[terms[i:] for i in range(3)]): <-- solution without a generator (works the same but has higher memory usage)
            for ngram in zip(*[islice(seq, i, len(terms)) for i, seq in enumerate(tee(terms, ngramLength))]): # <-- solution using a generator
                ngram = ' '.join(ngram)
                yield ngram
count_vect = CountVectorizer(analyzer=words_per_line)
token = text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
graph = tf.get_default_graph()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


@app.route('/')

def home():
  return render_template("ome.html")
#run_with_ngrok(app)
from keras.models import load_model
from keras.models import model_from_json
import json
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

'''with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')'''
#model=load_model('modrnn.h5')
models = {}
vect = pickle.load(open('vectorizer.pickle', 'rb'))

    # load the model
models['NB'] = pickle.load(open('classification.model', 'rb'))
models['RNN'] = load_model('modrnn.h5')
models['SVM'] = pickle.load(open('svm.model', 'rb'))
graph = tf.get_default_graph()

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        word = request.form.get('word')
        model_name=request.form.get('model')
        model = models[model_name]
        if model_name == 'NB':
           twt=vect.transform([word])
        elif model_name == 'SVM':
          twt=vect.transform([word])    
        else:
            twt = sequence.pad_sequences(loaded_tokenizer.texts_to_sequences([word]), maxlen=80, dtype='int32', value=0)
        with graph.as_default():
             j = model.predict(twt)[0]
             if j==1:
               k="positive"
             elif j==0:
               k="negative"
             else:
               k=j
        return '''<h1> The model used is {}</h1><br>
                <h1> Sentiment: {}</h1>
                '''.format(model_name,k) 
                                               
    return render_template("predict.html")

@app.route('/realtime', methods=["GET","POST"])
def realtime():
  s=[]
  per1=0
  per2=0
  cnt=0
  neg=0
  pos=0
  model=models['NB']
  tweetCount=0
  if request.method == 'POST':
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
               wait_on_rate_limit_notify=True)
    search = request.form.get('word')
    public_tweets = api.search(q=search, lang="te", count=100)
    for tweet in public_tweets:
              
                Tweet = tweet.text
                Tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',Tweet)
                Tweet = re.sub('@[^\s]+','TWITTER_USER',Tweet)
                Tweet = re.sub('[\s]+', ' ', Tweet)
                Tweet = re.sub(r'#([^\s]+)', r'\1', Tweet)
                Tweet = Tweet.strip('\'"')
                a = ':)'
                b = ':('
                Tweet = Tweet.replace(a,'')
                Tweet = Tweet.replace(b,'')
                tag = 'TWITTER_USER' 
                rt = 'RT'
                url = 'URL'
                Tweet = Tweet.replace(tag,'')
                tweetCount+=1
                if rt in Tweet:
                  continue
                Tweet = Tweet.replace(rt,'')
                Tweet = Tweet.replace(url,'')
                twt=vect.transform([Tweet])
                j = model.predict(twt)[0]
                if j==0:
                  neg=neg+1
                else:
                  pos=pos+1
                cnt=cnt+1

                s.append(Tweet) 
                s.append(j)
    per1=((float(pos)/cnt)*100)
    per2=((float(neg)/cnt)*100)
             
  return render_template('home.html',
                          tweets=s,per1=per1,per2=per2)                                       
  #api.user_timeline(search)
    
if __name__ == '__main__':
    app.run()