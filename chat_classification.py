import streamlit as st
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from NBmodel import get_prepared_csvs, get_trump_biden_data, get_train_data, text_process
import pandas_read_xml as pdx
from cam_chat_classification import add_features, construct_bag_of_words, \
                            get_batch_features, featurize_msg, prepare_msg, \
                            prepare_one_msg
from PIL import Image
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.metrics import confusion_matrix


model = MultinomialNB()
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Text classification")
st.header("By Cameron Abbot and Abraham Holleran")

st.write("We can use a Naive Bayes classifier to detect if a sentence is more likely\
 to be said by you or your friend. Simply download the \"SMS Backup & Restore\" app and\
         download the messages as an xml file. Then, write the path to that file here.")

st.write("You can either type in the url path to the xml file, or for convenience you can enter the ID of the Google Drive file. If you host the file on drive and turn link sharing on, the link will include the ID: drive.google.com/file/d/THIS_IS_THE_ID//view?usp=sharing")

xml_path = st.text_input("URL path to xml data file", "")

# Prevents running errors before user has entered data file URL
if(xml_path == ""):
    st.write("No data entered.")
    import sys
    sys.exit()

try:
    df_sms = pdx.read_xml("https://drive.google.com/uc?id=" + xml_path, encoding="utf8")
    name, df = add_features(df_sms) 
except:
    df_sms = pdx.read_xml(xml_path, encoding="utf8")
    name, df = add_features(df_sms)

df['person'] = df['person'].astype(int)
first = df['person'][0]
countt = [0,0]
who_spoke = [[],[]]
for i in range(len(df['person'])):
    if df['person'][i] == first:
        countt[0] += 1
        who_spoke[0].append(i)
    else:
        countt[1] += 1
        who_spoke[1].append(i)
size_limit = min(countt)

df['author'] = np.where(df['person'] == 2, 'me', name)
df['body'] = df['body'].astype(str)
df['body'] = df['body'].str.lower()


who_spoke = [who_spoke[0][:size_limit], who_spoke[1][:size_limit]]
loca = who_spoke[0] + who_spoke[1]
more_data = df.iloc[loca]


bag_of_words = construct_bag_of_words(more_data['body'])
batch_features = get_batch_features(more_data['body'], bag_of_words)

X_train, X_test, y_train, y_test = \
    train_test_split(batch_features, more_data['person'], test_size=0.2, random_state=42)

clf = Perceptron(n_iter_no_change=50)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)


st.write(f"After training, we have an accuracy report. In the following confusion matrix, \
{name} is 0 and you are 1. The columns are what's predicted, and the rows are what's true.\
 \n", confusion_matrix(y_test.tolist(), pred), '\n')
user_text = st.text_input("Text to classify goes here:", "I like math, \
 but fractions is where I draw the line.")

test_features = prepare_one_msg(user_text, bag_of_words)
potential_author = clf.predict(test_features)

if potential_author[0] == 1:
    potential_author = name
else:
    potential_author = 'you'


st.write(f"With {round(score, 2)}% accuracy on the test data, \
we predict that your message was said by {potential_author}.")



tb_im = Image.open('Trump-or-Biden.jpg')
st.image(tb_im, caption='Commander in Chief', use_column_width=True)
st.header("Trump/Biden classification")
st.write("Please input your data and your message to classify as Trump/Biden.")
user_tb_text = st.text_input("Text to classify goes here:", "Will you shut up, man")
#Classify as Trump/Biden by removing all the welker, wallace stuff.

first_debate, second_debate = get_prepared_csvs()
trump_biden_data = get_trump_biden_data(second_debate, first_debate)
X, y = get_train_data(trump_biden_data)
transformer = CountVectorizer(analyzer=text_process).fit(X)
all_train = transformer.transform(X)
model = model.fit(all_train, y)

user_tb_text = pd.Series(user_tb_text)
user_transformed = transformer.transform(user_tb_text)
pred_speaker = model.predict(user_transformed)
if pred_speaker[0] == 0:
    pred_speaker = "Biden"
elif pred_speaker[0] == 1:
    pred_speaker = "Trump"
st.write(f"With >80% accuracy, we predict that the message was said by {pred_speaker}.")

