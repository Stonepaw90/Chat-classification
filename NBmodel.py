import numpy as np
import pandas as pd
import scipy
import nltk
import string
import streamlit as st


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

nltk.download('wordnet')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
labelencoder = LabelEncoder()
@st.cache
def get_prepared_csvs():
    df = pd.read_csv('debate.csv')
    df1 = pd.read_csv('debate1.csv')
    df1.drop(df1.columns[[1]], axis = 1, inplace=True)
    df1 = df1.rename(columns={"text": "speech"})
    for i in range(len(df1['speaker'])):
      if df1['speaker'][i] == 'Chris Wallace':
        df1['speaker'][i] = 'WALLACE'
      if df1['speaker'][i] == 'Chris Wallace:':
        df1['speaker'][i] = 'WALLACE'
      if df1['speaker'][i] == 'Vice President Joe Biden':
        df1['speaker'][i] = 'BIDEN'
      if df1['speaker'][i] == 'President Donald J. Trump':
        df1['speaker'][i] = 'TRUMP'
    df.drop(df.columns[[0, 1]], axis = 1, inplace=True)
    return df1, df
@st.cache
def speaker_together(authors, speech):
  #Combines what each person said into one long string.
  combined_df = pd.DataFrame(columns = ['Speaker', 'Speech'])
  prev_speaker = authors[0]
  first_sentence = ''
  for i in range(len(authors)):
    speaker = authors[i]
    if speaker != prev_speaker:
      #We've hit a new speaker. Add what we had and save the new first text.
      combined_df = combined_df.append({'Speaker' : prev_speaker, 'Speech' : first_sentence}, ignore_index = True)
      prev_speaker = speaker
      first_sentence = speech[i]
    else:
      #print(i)
      first_sentence += speech[i]
  return combined_df
@st.cache
def get_trump_biden_data(df, df1):
    fixed_df = speaker_together(df['speaker'], df['speech'])
    fixed_df1=speaker_together(df1['speaker'], df1['speech'])
    fixed_df = fixed_df1.append(fixed_df, ignore_index=True)
    fixed_df['Speech'] = fixed_df['Speech'].str.replace('crosstalk', '')
    fixed_df.drop(fixed_df.loc[fixed_df['Speaker']=='WALLACE'].index, inplace=True)
    fixed_df.drop(fixed_df.loc[fixed_df['Speaker']=='WELKER'].index, inplace=True)
    return fixed_df
# Defining a module for Text Processing
@st.cache
def text_process(tex):
    # 1. Removal of Punctuation Marks
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    # 2. Lemmatisation
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    # 3. Stemming the words
    stemmer.stem(a)
    # 4. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('english')]


@st.cache
def get_train_data(fixed_df):
    y = fixed_df['Speaker']
    y = labelencoder.fit_transform(y)
    # 80-20 splitting the dataset (80%->Training and 20%->Validation)
    X = fixed_df['Speech']
    return X, y