import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import itemgetter
import json
from main import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path = './/Data//Emails'
dict_path = './/Data//English Words'
name_path = './/Data//English Names'
additional_path  = './/Data//Additional'

train_data = pd.read_csv(os.path.join(path,'emails.train.csv'))
test_data = pd.read_csv(os.path.join(path, 'emails.test.csv'))
all_data = pd.concat([train_data, test_data])
all_data = all_data.sort_values('id')
train_indices = train_data.id.values
test_indices = test_data.id.values

english_dict_file = os.path.join(dict_path, 'words_dictionary.json')
with open(english_dict_file,"r") as english_dictionary:
    english_words_dict = json.load(english_dictionary)
english_words_set = set(english_words_dict.keys())

english_fnames_file = os.path.join(name_path, 'popular-both-first.txt')
first_names = pd.read_csv(english_fnames_file, header=None)
english_fnames_list = first_names.iloc[:, 0]
english_fnames_list = [english_fnames_list[i].lower() for i in range(len(english_fnames_list))
                      if isinstance(english_fnames_list[i], str)]
english_fname_set = set(english_fnames_list)

english_lnames_file = os.path.join(name_path, 'census-dist-2500-last.csv')
last_names = pd.read_csv(english_lnames_file, header=None)
english_lnames_list = last_names.iloc[:, 0]
english_lnames_list = [english_lnames_list[i].lower() for i in range(len(english_lnames_list))
                      if isinstance(english_lnames_list[i], str)]
english_lname_set = set(english_lnames_list)

english_greetings_file = os.path.join(additional_path, 'greetings.txt')
greetings = pd.read_csv(english_greetings_file, header=None)
greetings_list = greetings.iloc[:, 0]
greetings_set = set(greetings_list)

indicator_file = os.path.join(additional_path, 'indicators.txt')
indicators = pd.read_csv(indicator_file, header=None)
indicators_list = indicators.iloc[:, 0]
indicators_set = set(indicators_list)

date_file = os.path.join(additional_path, 'days_months.txt')
dates = pd.read_csv(indicator_file, header=None)
dates_list = indicators.iloc[:, 0]
date_set = set(indicators_list)

#X = create_design_matrix(all_data, english_words_set, english_fname_set, english_lname_set,
#                         greetings_set, indicators_set, date_set)
X = create_bow_matrix(all_data)
Y = create_response_vector(all_data)

print(X.shape)
X_train = X[train_indices.tolist(), :]
Y_train = Y[train_indices.tolist()]
X_test = X[test_indices.tolist(), :]

np.save('xtrain', X_train)
np.save('ytrain', Y_train)
np.save('xtest', X_test)
np.save('test_indices', test_indices)