import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os


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


def get_all_words(text_list):
    local_word_list = []
    for i in range(len(text_list)):
        text = text_list[i]
        word_dict = word_split(text)
        for word in word_dict:
            local_word_list.append(word)
    return local_word_list
