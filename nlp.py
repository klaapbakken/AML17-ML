from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from main import *
from nltk.corpus import webtext, nps_chat, brown

path = './/Data//Emails'
train_data = pd.read_csv(os.path.join(path, 'emails.train.csv'))
test_data = pd.read_csv(os.path.join(path, 'emails.test.csv'))
all_data = pd.concat([train_data, test_data])
all_data = all_data.sort_values('id')
train_indices = train_data.id.values
test_indices = test_data.id.values

modified_stopwords = stopwords.words('english').copy()
modified_stopwords.append('Subject')

def tokenize(data):
    emails = data.text
    token_list = list()
    for i in range(len(emails)):
        token_list.append(word_tokenize(data.text.iloc[i]))
    return token_list

def tokenize_list(a_list):
    token_list = list()
    for i in range(len(a_list)):
        token_list.append(word_tokenize(a_list[i]))
    return token_list

def remove_stopwords(tokens_list, stopwords):
    words_list = list()
    for i in range(len(tokens_list)):
        words_list.append([word for word in tokens_list[i] if word not in stopwords])
    return words_list

def stem(words_list):
    stemmer = SnowballStemmer("english")
    stem_list = list()
    for i in range(len(words_list)):
        stem_list.append([stemmer.stem(word) for word in words_list[i]])
    return stem_list

def tfidf(emails):
    tfidf = TfidfVectorizer(stop_words=stopwords.words('english'),
                            strip_accents='unicode', sublinear_tf=True)
    return tfidf.fit_transform(emails).toarray()

def create_X(data):
    X = tfidf(data.text)
    return X

def create_Y(data):
    Y = data.spam.values
    return Y

def preprocess():
    tokens = tokenize(all_data)
    words = remove_stopwords(tokens, modified_stopwords)
    stem_list = stem(words)
    proc_em = [''.join(stem_list[i]) for i in range(len(stem_list))]
    processed_emails = pd.DataFrame(proc_em, columns=['text'])
    return processed_emails

def create_wordcloud(train_data):
    spam = [train_data.text.iloc[i] for i in range(len(train_data)) if train_data.spam.iloc[i] == 1]
    tokenized_spam = tokenize_list(spam)
    clean_spam = remove_stopwords(tokenized_spam, modified_stopwords)
    clean_spam_list = [' '.join(clean_spam[i]) for i in range(len(clean_spam))]
    clean_spam_string = ' '.join(clean_spam_list)
    top_spam = top_words(10, word_split(clean_spam_string)).keys()

    nspam = [train_data.text.iloc[i] for i in range(len(train_data)) if train_data.spam.iloc[i] == 0]
    tokenized_nspam = tokenize_list(nspam)
    clean_nspam = remove_stopwords(tokenized_nspam, modified_stopwords)
    clean_nspam_list = [' '.join(clean_nspam[i]) for i in range(len(clean_nspam))]
    clean_nspam_string = ' '.join(clean_nspam_list)
    top_non_spam = top_words(10, word_split(clean_nspam_string)).keys()

    wordcloud = WordCloud().generate(clean_spam_string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    print(list(top_spam))


X = create_X(all_data)
Y = create_Y(all_data)

X_train = X[train_indices.tolist(), :]
Y_train = Y[train_indices.tolist()]
X_test = X[test_indices.tolist(), :]

np.save('xtrain', X_train)
np.save('ytrain', Y_train)
np.save('xtest', X_test)
np.save('test_indices', test_indices)

