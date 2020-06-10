from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn import naive_bayes
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pandas as pd
import re
import string

trainDF = pd.read_csv('totaldtl.csv',encoding='UTF-8'
                   )
trainDF.columns = ['text', 'label']
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+ string.punctuation +']', '', text)
    return text_nopunct
trainDF['text'] =trainDF['text'].apply(lambda x: remove_punct(x))
j=[]
for s in trainDF['text']:
    s = re.sub('[\u200c]', '', s)
    s = re.sub('[\u200b]', '', s)
    s = re.sub('[A-Za-z0-9]','',s)
    j.append(s)
trainDF['text']= pd.Series(j) 
from nltk import word_tokenize
tokens=[]
for sen in trainDF.text:
    tokens.append(word_tokenize(sen))
    
#tokens = [word_tokenize(sen) for sen in trainDF.text]
ind=[]
for i in tokens:
    for j in i:
        ind.append(j)
k=[]
def removeStopWords(tokens): 
    return [word for word in tokens if word not in k]
filtered_words = [removeStopWords(sen) for sen in tokens]
trainDF['text'] = [' '.join(sen) for sen in filtered_words]
# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('cc.te.300.vec',encoding='UTF-8')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')


# create a tokenizer 
token = text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=80)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=80)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_length=max([len(s.split()) for s in trainDF['text']])
num_words=len(word_index)+1

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,GRU,Bidirectional,GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.layers import Dropout
from keras import layers
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
EMBEDDING_DIM=300
model=Sequential()
embedding_layer=Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=80,trainable=False)

model.add(embedding_layer)



#model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(1,activation='sigmoid'))
#model.add(GRU(units=12,dropout=0.2,recurrent_dropout=0.2))
model.add(LSTM(150, dropout=0.3, recurrent_dropout=0.3))
#model.add(LSTM(128))
#model.add(Dropout(0.3))
#model.add(Dense(64, activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
adam=Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

model.fit(train_seq_x, train_y, epochs=10, validation_split=0.2, shuffle=True, batch_size=32)

score = model.evaluate(valid_seq_x, valid_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save the model
model.save("modrnn.h5")

import pickle
 
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)
# serialize model to JSON
import json

# lets assume `model` is main model 
'''model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("model_weights.h5")'''

print("Saved model to disk")