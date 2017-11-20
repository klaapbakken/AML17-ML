import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import itemgetter


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


def text_dict_check(text_dict, english_dict_set):
    correct_words = 0
    incorrect_words = 0
    text_set = set(text_dict.keys())
    for word in text_set:
        if word in english_dict_set:
            correct_words += 1
    return correct_words / len(text_dict)


def create_design_matrix(data, english_dict_set):
    features = 3
    observations = len(data)
    X = np.zeros((observations, features))
    for i in range(observations):
        mail = data.text.iloc[i]
        # Number of unique words
        d = word_split(mail)
        X[i, 0] = len(d) / sum(d.values())
        # Proportion correct words
        X[i, 1] = text_dict_check(d, english_dict_set)
    return X


def create_response_vector(data):
    Y = np.array(data.spam)
    Y[Y == 0] = -1
    return Y


def data_split(data):
    test_indices = np.random.choice(len(data), len(data) // 5)
    train_indices = np.array([i for i in range(len(data)) if i not in test_indices])
    Y = create_response_vector(data)
    X = create_design_matrix(data, english_words_set)
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    return X_train, X_test, Y_train, Y_test
