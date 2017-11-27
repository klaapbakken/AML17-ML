import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import itemgetter
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def word_split(text):
    word_dict = {}
    current_word = str()
    for i in range(len(text)):
        if text[i].isalpha():
            current_word += text[i]
            continue
        else:
            if current_word:
                if current_word in word_dict:
                    word_dict[current_word] += 1
                else:
                    word_dict[current_word] = 1
                current_word = str()
            else:
                continue
    return word_dict


def local_words(text_list):
    local_word_list = []
    for i in range(len(text_list)):
        text = text_list.iloc[i]
        word_dict = word_split(text)
        for word in word_dict:
            local_word_list.append(word)
    return local_word_list


def local_word_dict(text_list):
    local_word_dict = {}
    for i in range(len(text_list)):
        word_dict = word_split(text_list.iloc[i])
        local_word_dict = {word: local_word_dict.get(word, 0) + word_dict.get(word, 0) for word in
                           set(local_word_dict) | set(word_dict)}
    return local_word_dict


def top_words(n, word_dict):
    assert n < len(word_dict)
    top_dict = {}
    sorted_keys = sorted(word_dict, key=word_dict.get, reverse=True)
    for i in range(n):
        top_dict[sorted_keys[i]] = word_dict[sorted_keys[i]]
    return top_dict


def personal_names_check(text_dict, english_name_set):
    personal_names = 0
    text_set = set(text_dict.keys())
    for word in text_set:
        if word in english_name_set:
            personal_names += 1
    return personal_names


def text_dict_check(text_dict, english_dict_set):
    correct_words = 0
    incorrect_words = 0
    text_set = set(text_dict.keys())
    for word in text_set:
        if word in english_dict_set:
            correct_words += 1
    return correct_words / len(text_dict)


def create_design_matrix(data, ew_set, ef_set, el_set, g_set, i_set, d_set):
    bow = bag_of_words(data.text)
    observations = len(data)
    features = 9 + bow.shape[1]
    X = np.zeros((observations, features))
    for i in range(observations):
        mail = data.text.iloc[i]
        d = word_split(mail)
        # Proportion of unique words
        X[i, 0] = len(d) / sum(d.values())
        # Proportion correct words
        X[i, 1] = text_dict_check(d, ew_set)
        # Number of personal names
        X[i, 2] = personal_names_check(d, ef_set)
        # Number of last names
        X[i, 3] = personal_names_check(d, el_set)
        # Number of greetings
        X[i, 4] = personal_names_check(d, g_set)
        # Number of indicators
        X[i, 5] = personal_names_check(d, i_set)
        # Presence of days, months, holidays
        X[i, 6] = personal_names_check(d, d_set)
        # Number of words
        X[i, 7] = sum(d.values())
        # Average word length
        av_len = 0
        for word in d:
            av_len += len(word) / sum(d.values())
        X[i, 8] = av_len
        X[i, 9:] = np.array(bow[i, :])
    return X


def create_bow_matrix(data):
    bow = bag_of_words(data.text)
    observations = len(data)
    features = bow.shape[1]
    X = np.zeros((observations, features))
    for i in range(observations):
        X[i, :] = np.array(bow[i, :])
    return X


def create_response_vector(data):
    Y = data.spam.values
    return Y


def bag_of_words(emails):
    CV = CountVectorizer()
    return CV.fit_transform(emails).toarray()


def tfidf(emails):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(emails).toarray()
