import string #removing punctuation

import pandas as pd
import numpy as np
import nltk #download

from nltk.corpus import stopwords #remove stopwords that are part of email
from nltk.stem.porter import PorterStemmer #reduce word down to stem

from sklearn.feature_extraction.text import CountVectorizer #take word stem and vectorise them by counting tokens, so we have a numerical representation taht we can sub into a model
from sklearn.model_selection import train_test_split #
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

df = pd.read_csv('spam_ham_dataset.csv')

df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

stemmer = PorterStemmer()
#stemmer.stem('Running') #becomes run or sophisticated becomes sophist
corpus = [] #transformed version of email

stopword_set = set(stopwords.words('english'))

for i in range(len(df)):
     text = df['text'].iloc[i].lower() #takes emails and makes lowercase
     text = text.translate(str.maketrans('','', string.punctuation)).split() #we used a third parameter here in maketrans but if you used just to e.g. maketrans('abc', 'xyz') all abc will be turned into xyz
     text = [stemmer.stem(word) for word in text if word not in stopword_set]
     text = ' '.join(text)
     corpus.append(text)

# print(df.text.iloc[0], corpus[0]) #test that loop is working

vectoriser = CountVectorizer() #turns the text into arrays of numbers

X = vectoriser.fit_transform(corpus).toarray()
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2) #test_size = 0.2 means that you are setting 20% of test data to test set and 80% to train set

clf = RandomForestClassifier(n_jobs = -1) #-1 so it uses all cpu cores

clf.fit(X_train, y_train)

#print(clf.score(X_test, y_test))

email_to_classify = df.text.values[10]

print(email_to_classify)

email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in text if word not in stopword_set]
email_text = ' '.join(text)

email_corpus = [email_text]

X_email = vectoriser.transform(email_corpus) #not fit transofrm here because its alredy fitted we don't want to refit again

print(clf.predict(X_email))

print(df.label_num.iloc[10])
