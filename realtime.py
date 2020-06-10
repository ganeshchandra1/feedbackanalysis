import codecs
import tweepy
import time
import  sys
import csv
import json
from tweepy import OAuthHandler
import nltk
import re


#required tokens for working with data from twitter using Tweepy
access_token = "103589925-PYiNRi6sAoSAFCau7Q5zDAqF7Kt8WwsK5EunWL3I"
access_token_secret = "u8N1nS93eN5npmtBOAxCwJgZE0W4wPCNe1CEuCB9lEIys"
consumer_key = "IIFBxSZv8YnhRJuvvDJVkR4ht"
consumer_secret = "zou3XOVMDXp9esingDNEowUeEPmTKkY4daZGYdmalovyd9JCxr"
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,
               wait_on_rate_limit_notify=True)
#Authenticating with given tokens
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)
def twitter_miner(query):
    tweetCount=0
    maxTweets=200
    neg=0
    pos=0
    cnt=0
    file1 = open('conf.txt', 'r',encoding="utf-8") 
    file2 = open('ou.txt', 'w',encoding="utf-8") 
    Lines = file1.readlines()
    while tweetCount < maxTweets:
        try:
           

            
            newTweets=api.search(q=query, lang="te", count=200)
            for tweet in newTweets:
                # printing only tweet
                #print (tweet.text)
                #print "####************************####"
                Tweet = tweet.text
                #Convert www.* or https?://* to URL
                Tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',Tweet)
                
                #Convert @username to User
                Tweet = re.sub('@[^\s]+','TWITTER_USER',Tweet)
                
                #Remove additional white spaces
                
                Tweet = re.sub('[\s]+', ' ', Tweet)
                
                #Replace #word with word Handling hashtags
                Tweet = re.sub(r'#([^\s]+)', r'\1', Tweet)
                
                #trim
                Tweet = Tweet.strip('\'"')
                
                #Deleting happy and sad face emoticon from the tweet 
                a = ':)'
                b = ':('
                Tweet = Tweet.replace(a,'')
                Tweet = Tweet.replace(b,'')
                
                #Deleting the Twitter @username tag and reTweets
                tag = 'TWITTER_USER' 
                rt = 'RT'
                url = 'URL'
                Tweet = Tweet.replace(tag,'')
                tweetCount+=1
                if rt in Tweet:
                  continue
                Tweet = Tweet.replace(rt,'')
                Tweet = Tweet.replace(url,'')
                """l1= []
                for word in l:
                  if not word.isalpha() and not word[0]=='#' and not word[0]=='@' and not word[0]=='h':
                     l1.append(word)"""
                print(Tweet)
                file2.writelines(Tweet)
                res=fn1(Tweet)
                print(res)
                file2.writelines(res+"\n")
            
                if res == "negative":
                  neg=neg+1
                elif res == "positive":
                  pos=pos+1
                cnt=cnt+1
                #csvWriter.writerow([tweet.text])

        except tweepy.TweepError as e:
            print("some error : " + str(e))
            print("retrying in 20 seconds")
            time.sleep(20)
    
    per1=((float(pos)/cnt)*100)
    per2=((float(neg)/cnt)*100)
    return per1,per2
    print("POSITIVE {0}%".format(per1))
    print("NEGATIVE {0}%".format(per2))
def mainmethod(keyword):
  print("\nFetching'{0}':\n".format(keyword))
  per1,per2=twitter_miner(keyword)
  return per1,per2
import nltk
import codecs
# training data set 
# positive tweets datset
pos_tweets = codecs.open("posamazon.txt",'r','utf-8')

# negative tweets dataset
neg_tweets =  codecs.open("negamazon.txt",'r','utf-8')
# merging the two list in one big training tuple and filtering two letters words 
tweets = []
sentiment = "positive"
for words in pos_tweets.readlines():
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

sentiment = "negative"
for words in neg_tweets.readlines():
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_tweets(tweets))
# building a feature extractor
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)

    return features

# building the training set
training_set = nltk.classify.apply_features(extract_features, tweets)

# training the classifier 
classifier = nltk.classify.NaiveBayesClassifier.train(training_set)

def fn1(tweet):
	#print tweet.split()
    return classifier.classify(extract_features(tweet.split()))


from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QLabel , QPlainTextEdit
from PyQt5.QtGui import QIcon , QColor
from PyQt5.QtCore import pyqtSlot , QSize,Qt , QRect

import sys
class App(QMainWindow):
 
    def __init__(self):
        super(App,self).__init__()
        self.title = 'Sentiment Analysis of Telugu Tweets'
        self.left = 10
        self.top = 10
        self.width = 350
        self.height = 400

        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        #create textbox for displaying
        self.DisplayTextbox = QPlainTextEdit(self)
        self.DisplayTextbox.setLineWrapMode(200)
        self.DisplayTextbox.move(20, 20)
        self.DisplayTextbox.resize(280,200)
       

        # Create textbox fro entering
        self.EnterTextbox = QLabel(self)
        self.EnterTextbox = QLineEdit(self)
        self.EnterTextbox.move(20, 300)
        self.EnterTextbox.resize(280,40)

 
        # connect enter textbox to function on_click
        self.EnterTextbox.returnPressed.connect(self.on_click)

        self.show()
 
    @pyqtSlot()

    def on_click(self):
        
        textfromgui = self.EnterTextbox.text()

        pos,neg=mainmethod(textfromgui)
        self.DisplayTextbox.setPlainText("POSISTIVE = "+str(pos)+"%\nNEGATIVE = "+str(neg)+"%")
 
 
def daba():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())