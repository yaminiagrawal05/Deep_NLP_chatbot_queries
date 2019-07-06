import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data = pd.read_csv('Sheet_1.csv')    #importing the dataset


corpus = []        # creating an object containing preprocessed data
for i in range(0,80):
    review = re.sub('[^a-zA-Z]',' ',data['response_text'][i])
    review = review.lower()
    review =review.split()
    ps =PorterStemmer()
    review = [ps.stem(word) for word in review if not word in (set(stopwords.words('english')))]
    review = ' '.join(review)
    corpus.append(review)
    
#bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X  = cv.fit_transform(corpus).toarray()

y=data.iloc[:, 1].values 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

model = Sequential()
model.add(Dense(264, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#y_train = np_utils.to_categorical(y_train, num_classes=1)
#y_test = np_utils.to_categorical(y_test, num_classes=1)

model.fit(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(x=X_test, y=y_test, batch_size=None, verbose=1, sample_weight=None)

L  = 'My friend tried to cut open his wrist'
review = re.sub('[^a-zA-Z]',' ',L)
review = review.lower()
review =review.split()
ps =PorterStemmer()
review = [ps.stem(word) for word in review if not word in (set(stopwords.words('english')))]
review = ' '.join(review)
L  = cv.fit_transform(corpus).toarray()

M =model.predict(L, batch_size=None, verbose=0, steps=None)


