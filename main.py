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
