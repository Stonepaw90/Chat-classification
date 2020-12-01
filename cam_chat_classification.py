# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pandas_read_xml as pdx
from pandas_read_xml import auto_separate_tables
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import string
#from wordcloud import WordCloud
import PIL
import itertools
#import matplotlib.pyplot as plt
import re
import itertools
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import streamlit as st
from time import time


# data path to xml file
df_sms = pdx.read_xml("G:/College/Machine Learning/data/addison.xml", encoding="utf8")
def add_features(df_sms):
# value is an ordered dictionary, which can then be converted back to a dataframe
    dictionary = df_sms.at['sms','smses']
    df_sms = pd.DataFrame(dictionary)
    contact_name = df_sms.at[0,'@contact_name'].lower()
    df_sms = df_sms[['@date','@type','@body']]
    df_sms.columns = ['date', 'person', 'body']
    return contact_name, df_sms




the_labels = ['one','two']
stop_words =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such','no','nor','not',
            'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
            'now','uses','use','using','used','one','also']

def preprocess(data):
    messages_tokens = []
    for message in data:
        message = message.lower() #Convert to lower-case words
        raw_word_tokens = re.findall(r'(?:\w+)', message,flags = re.UNICODE) #remove puntuaction
        word_tokens = [w for w in raw_word_tokens if not w in stop_words] # do not add stop words
        messages_tokens.append(word_tokens)
    return messages_tokens #return all tokens

def wordcloud_stuff(frames, complete_data):
    for messages,label in zip(frames,the_labels):
        raw_str = complete_data.loc[label].str.cat(sep=',')
        wordcloud = WordCloud( max_words=1000,margin=0).generate(raw_str)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    for messages,label in zip(frames,the_labels):
        tokenized_messages = preprocess(messages) #apply the preprocess step
        messages = list(itertools.chain(*tokenized_messages))
        text_messages = " ".join(messages)
        wordcloud = WordCloud( max_words=1000,margin=0).generate(text_messages)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()



def construct_bag_of_words(data):
    corpus = preprocess(data)
    bag_of_words = {}
    word_count = 0
    for sentence in corpus:
        for word in sentence:
            if word not in bag_of_words:  # do not allow repetitions
                bag_of_words[word] = word_count  # set indexes
                word_count += 1

    # print(dict(Counter(bag_of_words).most_common(5)))
    return bag_of_words  # index of letters

#bag_of_words = construct_bag_of_words(complete_data)


def featurize(sentence_tokens,bag_of_words):
    sentence_features = [0 for x in range(len(bag_of_words))]

    for word in sentence_tokens:
        index = bag_of_words[word]
        sentence_features[index] +=1
    return sentence_features



def get_batch_features(data,bag_of_words):
    batch_features = []
    messages_text_tokens = preprocess(data)
    for message_text in messages_text_tokens:
        feature_message_text = featurize(message_text,bag_of_words)
        batch_features.append(feature_message_text)
    return batch_features

#batch_features = get_batch_features(complete_data,bag_of_words)








def featurize_msg(msg,bag_of_words):
    msg_features = [0 for x in range(len(bag_of_words))]

    for word in msg:
      try:
        index = bag_of_words[word]

        msg_features[index] +=1
      except:
        print(word)
        pass
    return [msg_features]

def prepare_msg(data,bag_of_words):
    batch_features = []
    # data is a string, needs to be an array to separate words correctly
    data = [data]
    messages_text_tokens = preprocess(data)
    for message_text in messages_text_tokens:
        feature_message_text = featurize_msg(message_text,bag_of_words)
        batch_features.append(feature_message_text)
        # print("text: ", message_text)
    return batch_features[0]
    
def prepare_one_msg(data,bag_of_words):
    batch_features = []
    # data is a string, needs to be an array to separate words correctly
    data = [data]
    messages_text_tokens = preprocess(data)
    for message_text in messages_text_tokens:
        feature_message_text = featurize_msg(message_text,bag_of_words)
        batch_features.append(feature_message_text)
        # print("text: ", message_text)
    return batch_features[0]