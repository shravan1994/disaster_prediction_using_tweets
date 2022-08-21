import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, Conv1D, MaxPool1D, Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import tensorflow_text as text
import re
import contractions
import tensorflow_hub as hub



MODEL_PATH = 'saved_model/best_bert_model.h5'
TRAINED_MODEL = None

def load_bert():
    bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True)

    return bert_preprocess_model, bert_layer


def load_model_architecture():
    bert_preprocess_model, bert_layer = load_bert()
    input_layer1 = Input(shape=(), dtype=tf.string, name='text-layer')
    text_preprocessed = bert_preprocess_model(input_layer1)
    bert_outputs = bert_layer(text_preprocessed)
    pooled_out = bert_outputs['pooled_output']
    dropout_1 = Dropout(0.1)(pooled_out)

    dense1 = tf.keras.layers.Dense(128,activation='relu')(dropout_1)
    dense2 = tf.keras.layers.Dense(64,activation='relu')(dense1)
    dropout_2 = tf.keras.layers.Dropout(0.2)(dense2)

    output = Dense(1, activation='sigmoid')(dropout_2)

    model = Model(input_layer1, output)

    return model


def replace_byte_chars(tweet):
  tweet = re.sub(r"\x89Ûªs", '\'s', tweet)
  tweet = re.sub(r"\x89Û_", "", tweet)
  tweet = re.sub(r"\x89ÛÒ", "", tweet)
  tweet = re.sub(r"\x89ûó", "", tweet)
  tweet = re.sub(r"\x89ÛÏ", "", tweet)
  tweet = re.sub(r"\x89Û÷", "", tweet)
  tweet = re.sub(r"\x89Û", "", tweet)
  tweet = re.sub(r"\x89Û\x9d", "", tweet)
  tweet = re.sub(r"\x89Û¢", "", tweet)
  tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
  tweet = re.sub(r"åÊ", " ", tweet)
  tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
  tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
  tweet = re.sub(r"Ì©", "e", tweet)
  tweet = re.sub(r"å¨", "", tweet)
  tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
  tweet = re.sub(r"åÇ", "", tweet)
  tweet = re.sub(r"å£3million", "3 million", tweet)
  tweet = re.sub(r"åÀ", "", tweet)
  
  return tweet


def do_decontractions(tweet):
  tweet = contractions.fix(tweet)
  return tweet


def clean_tweet(tweet):
  tweet = tweet.lower()
  tweet = replace_byte_chars(tweet)
  tweet = do_decontractions(tweet)
  # removing urls
  tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
  # Words with punctuations and special characters
  tweet = re.sub(r"[\"#$%&'()*+,\-.\/:;<=>@[\]^_`{|}~]", "", tweet)
  # adding space in front of ? and !
  tweet = re.sub(r"([?!]+)", r" \1", tweet)
  tweet = re.sub(r"\s+", " ", tweet)
  tweet = tweet.strip()
  return tweet


def model_predict(X):
    global TRAINED_MODEL

    X = pd.Series([X]) 
    X = X.apply(lambda x: clean_tweet(x))

    if not TRAINED_MODEL:
        TRAINED_MODEL = load_model_architecture()
        TRAINED_MODEL.load_weights(MODEL_PATH)

    target_predicted = TRAINED_MODEL.predict(X)

    return 1 if target_predicted[0] > 0.5 else 0