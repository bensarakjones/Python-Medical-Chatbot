from ctypes.wintypes import WORD
import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer
intents = json.loads(open('intents.json').read())

words = []

classes = []

documents = []

ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for patterns in intents['Patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.append(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
            print(documents)
