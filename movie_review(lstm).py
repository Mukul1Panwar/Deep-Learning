!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import pandas as pd
import numpy as np

df = pd.read_csv("/content/imdb-dataset-of-50k-movie-reviews.zip")

df.head()

df['sentiment'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['sentiment']=le.fit_transform(df['sentiment'])

df['sentiment'].value_counts()

df['review']

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM,Embedding,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=50000)

tokenizer.fit_on_texts(df['review'])

sequence = tokenizer.texts_to_sequences(df['review'])

len(sequence)

for sen in sequence:
  print(len(sen))

max(len(sen) for sen in sequence)

from tensorflow.keras.preprocessing.sequence import pad_sequences

x = pad_sequences(sequence,maxlen=200)

x

y = df['sentiment']

from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(
    x,y,test_size=0.2,random_state=42
)

Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape

from tensorflow.keras.regularizers import l1,l2

model = Sequential()

model.add(Embedding(input_dim=50000,output_dim=100,input_length=200)) #Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)
model.add(LSTM(128,dropout=0.2))
model.add(Dense(64,kernel_regularizer=l1(0.001)))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(Xtrain,Ytrain,epochs=4,batch_size=64,validation_split=0.2)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'],label='training')
plt.plot(history.history['val_loss'],label='testing')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label='training')
plt.plot(history.history['val_accuracy'],label='testing')
plt.legend()
plt.show()

sentence = input("enter the review : ")
sentence1 = tokenizer.texts_to_sequences([sentence])
sen_pad = pad_sequences(sentence1,maxlen=200)
prediction = model.predict(sen_pad)[0][0]

print(prediction)

if prediction >=0.5 :
  print("Positive")
else:
  print("Negative")

