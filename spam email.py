import string #removing punctuation

import pandas as pd
import numpy as np
import nltk #download
import unicodedata

from nltk.corpus import stopwords #remove stopwords that are part of email
from nltk.stem.porter import PorterStemmer #reduce word down to stem

from sklearn.feature_extraction.text import CountVectorizer #take word stem and vectorise them by counting tokens, so we have a numerical representation taht we can sub into a model
from sklearn.model_selection import train_test_split #
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

df = pd.read_csv('spam_ham_dataset.csv')

df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

stemmer = PorterStemmer()
#stemmer.stem('Running') #becomes run or sophisticated becomes sophist
corpus = [] #transformed version of email

stopword_set = set(stopwords.words('english'))

for i in range(len(df)):
     text = df['text'].iloc[i].lower() #takes emails and makes lowercase
     text = ''.join([char if unicodedata.category(char) not in ['Pc','Po','So','Sm','Sc'] else ' ' for char in text.lower()]).split() #we used a third parameter here in maketrans but if you used just to e.g. maketrans('abc', 'xyz') all abc will be tunred into xyz
     text = [stemmer.stem(word) for word in text if word not in stopword_set]
     text = ' '.join(text)
     corpus.append(text)

# print(df.text.iloc[0], corpus[0]) #test that loop is working

vectoriser = CountVectorizer(min_df=3) #turns the text into arrays of numbers

X = vectoriser.fit_transform(corpus).toarray()
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2) #test_size = 0.2 means that you are setting 20% of test data to test set and 80% to train set

# clf = RandomForestClassifier(
#     n_estimators=50,          # Fewer trees
#     max_depth=5,              # Much shallower trees
#     min_samples_split=20,     # Need more samples to split
#     min_samples_leaf=10,      # Minimum samples in each leaf
#     max_features='sqrt',      # Use fewer features per tree
#     class_weight='balanced',  # Handle class imbalance
#     random_state=42,
#     n_jobs=-1
#  ) #-1 so it uses all cpu cores

clf = MultinomialNB(alpha=5.0)

clf.fit(X_train, y_train)

print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# See if model is overfitting
if clf.score(X_train, y_train) > 0.95:
    print("Model might be overfitting!")

# print(df['label_num'].value_counts())
# print(f"Spam percentage: {df['label_num'].mean():.1%}")

#print(clf.score(X_test, y_test))


# email_to_classify = df.text.values[10]

# print(email_to_classify)

# email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
# email_text = [stemmer.stem(word) for word in email_text if word not in stopword_set]
# email_text = ' '.join(email_text)

# email_corpus = [email_text]

# X_email = vectoriser.transform(email_corpus) #not fit transofrm here because its alredy fitted we don't want to refit again

# clf.predict(X_email) #1 is spam, 0 is ham

# df.label_num.iloc[10]

# email_to_classify = input('Please enter email that you would like to check.')

# email_to_classify = email_to_classify.lower()
# email_text = ''.join([char if unicodedata.category(char) not in ['Pc','Po','So','Sm','Sc'] else ' ' for char in email_to_classify.lower()]).split()
# email_text = [stemmer.stem(word) for word in email_text if word not in stopword_set]
# email_text = ' '.join(email_text)

# email_corpus = [email_text]

# X_email = vectoriser.transform(email_corpus)

# print(clf.predict(X_email))

test_email = "Dear all,I appreciate this might seem very early indeed, but I am beginning to think about the Choral Society concert in Michaelmas and the plan is to do Verdi's epic Requiem. This requires a substantial orchestra but there is plenty to do! To this end, I am writing to ask for you to play in this orchestra which would mean a tuesday rehearsal period 8 for the tuesdays until 30th November (10 rehearsals). The orchestra for this performance will end up being about 45 players, with many external but I would really wish for you to have this opportunity - it is quite a rare feat really.I know that you have to sign up for next term at some point in this term and so, whilst I appreciate this is not a forced thing, I would be grateful if you would take this opportunity to be part of a massive musical project. Choir will be about 120.I am sure there might be some clashes, but the reality is you would have to attend most of these rehearsals to get near the part, something I believe you can all do if you attend these tuesdays next term.If you have any queries over this or questions, I hope you know my door is open. If this is something you don't want to do/can't do, please do come and speak to me to see if there is compromise.All best,Mr Stafford"
processed = ''.join([char if unicodedata.category(char) not in ['Pc','Po','So','Sm','Sc'] else ' ' for char in test_email.lower()]).split()
processed = [stemmer.stem(word) for word in processed if word not in stopword_set]
final_processed = ' '.join(processed)
print("Your email becomes:", final_processed)

email_corpus = [final_processed]

X_email = vectoriser.transform(email_corpus)

print(clf.predict(X_email))

prediction = clf.predict(X_email)[0]
probability = clf.predict_proba(X_email)[0]
print(f"Prediction: {prediction}")
print(f"Spam probability: {probability[1]:.2%}")
print(f"Ham probability: {probability[0]:.2%}")

# Your first email that worked (short and simple)
email1 = "Dear Mr Franklin,Sorry for not getting back to you sooner. I hope you have had an amazing summer. I would be happy to help.Kind regards,Charlie Martin"

# Your second email that fails (long and formal) 
email2 = "Dear all,I appreciate this might seem very early indeed, but I am beginning to think about the Choral Society concert..."  # your full email

def process_email(email_text):
    processed = ''.join([char if unicodedata.category(char) not in ['Pc','Po','So','Sm','Sc'] else ' ' for char in email_text.lower()]).split()
    processed = [stemmer.stem(word) for word in processed if word not in stopword_set]
    return processed

words1 = set(process_email(email1))
words2 = set(process_email(email2))

print("Words ONLY in the problem email:")
problem_words = words2 - words1
print(list(problem_words)[:20])

feature_names = vectoriser.get_feature_names_out()
print(f"Total vocabulary size: {len(feature_names)}")
print("Sample features:", feature_names[:20])

problem_words = ['earli', 'choral', 'appreci', 'might', 'concert', 'think', 'begin', 'inde', 'societi', 'seem']

# Create processed corpus for analysis
all_processed_texts = []
for i in range(len(df)):
    text = ''.join([char if unicodedata.category(char) not in ['Pc','Po','So','Sm','Sc'] else ' ' for char in df['text'].iloc[i].lower()]).split()
    text = [stemmer.stem(word) for word in text if word not in stopword_set]
    all_processed_texts.append(' '.join(text))

# Check word frequencies
for word in problem_words:
    spam_count = sum(1 for i, text in enumerate(all_processed_texts) if word in text and df.iloc[i]['label_num'] == 1)
    ham_count = sum(1 for i, text in enumerate(all_processed_texts) if word in text and df.iloc[i]['label_num'] == 0)
    
    if spam_count + ham_count > 0:
        spam_ratio = spam_count / (spam_count + ham_count)
        print(f"'{word}': Spam={spam_count}, Ham={ham_count}, Spam ratio={spam_ratio:.2%}")