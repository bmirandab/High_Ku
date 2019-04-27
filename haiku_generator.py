#!/usr/bin/env python
# coding: utf-8

# In[9]:


import random
import sys
import os
import keras
import pyphen
import re
import nltk
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from keras.backend import clear_session
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from nltk.corpus import words, wordnet
nltk.download('words')

with open('text.pkl', 'rb') as t:
    text = pickle.load(t)
    
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    np.seterr(divide = 'ignore') 
    return np.argmax(probas)


def haiku_generator(text):
    clear_session()

    seq_len = 50
    step = 3
    sentences = []
    next_chars = []

    for i in range(0, len(text) - seq_len, step):
        sentences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])

    chars = sorted(list(set(text)))
    char_indices = dict((char, chars.index(char)) for char in chars)
    
    n_chars = len(text)
    n_vocab = len(chars)
    n_sentences = len(sentences)
    
    model = load_model("lstm4_weights-improvement-01-1.0284.hdf5")
    
    start_index = random.randint(0, n_chars - seq_len - 1)
    generated_text = text[start_index: start_index + seq_len]

    for temperature in [0.2]:
        haiku = []
        for i in range(150):
            sampled = np.zeros((1, seq_len, n_vocab))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            haiku.append(next_char)

        haiku_gen = "".join(haiku)
        
    haiku = haiku_gen.split()
    en_haiku = [w for w in haiku if w in words.words()]
    
    dic = pyphen.Pyphen(lang='en')
    haiku_syllables =[]
    haiku_syllables = [dic.inserted(w) for w in en_haiku]
    
    syllables=[]
    for w in haiku_syllables:
        syllables_count = w.split('-')
        syllables.append([w, len(syllables_count)])
        
    line_count = {"line1": 0, "line2": 0, "line3": 0}
    haiku_final = {"line1": [], "line2": [], "line3": []}


    for w in syllables:
        if w[1] + line_count["line1"] <= 5:  
            haiku_final["line1"].append(w[0])
            line_count["line1"] = w[1] + line_count["line1"]
        elif w[1] + line_count["line2"] <= 7:
            haiku_final["line2"].append(w[0])
            line_count["line2"] = w[1] + line_count["line2"]
        elif w[1] + line_count["line3"] <= 5:
            haiku_final["line3"].append(w[0])
            line_count["line3"] = w[1] + line_count["line3"]

    lines = [" ".join(haiku_final['line1']), " ".join(haiku_final['line2']), " ".join(haiku_final['line3']) ]
    haiku_printable = "\n".join(lines)
    haiku_printable = haiku_printable.replace('-', '')
    print(haiku_printable)


# In[11]:


haiku_generator(text)


# In[ ]:




