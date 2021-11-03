# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:16:55 2021

@author: jkellogg
"""
import nltk
import nltk.corpus
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader
import random
random.seed(42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import plotly.io as pio

import pandas as pd
import numpy as np
import string

#pd.set_option("display.max_rows",500)

import warnings
warnings.filterwarnings('ignore')

#from nltk.sentiment import SentimentIntensityAnalyzer as sia
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import requests
import io   

president_list = ['Harry S. Truman',
'Richard M. Nixon',
'Dwight D. Eisenhower',
'John F. Kennedy',
'Lyndon B. Johnson',
'Ronald Reagan',
'Gerald Ford',
'Jimmy Carter',
'George H. W. Bush',
'Bill Clinton',
'George W. Bush',
'Barack Obama']

all_speechs_data = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/presidential_speeches.csv" 
obama_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/obama.csv" 
bush43_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/bush43.csv" 
bush41_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/bush41.csv"
clinton_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/clinton.csv" 
reagan_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/reagan.csv" 
carter_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/carter.csv" 
eisenhower_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/eisenhower.csv" 
ford_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/ford.csv" 
johnson_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/johnson.csv" 
kennedy_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/kennedy.csv" 
nixon_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/nixon.csv" 
truman_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/truman.csv" 
trump_rating = "https://raw.githubusercontent.com/kelloggjohnd/Final_project/main/data/approval_ratings/trump.csv" 

def data_pull(url):
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return df

def rolling_change(df):
    df['Approve_rolling']= df['Approving'].shift(-1).rolling(2, 2).median().shift(-1)
    df['Disapprove_rolling'] = df['Disapproving'].shift(-1).rolling(2, 2).median().shift(-1)
    df['Unsure_rolling'] = df['Unsure/NoData'].shift(-1).rolling(2, 2).median().shift(-1)
    df['Approve_change'] = (df ['Approve_rolling'] - df['Approving'])/df ['Approve_rolling'] *100
    df['Disapprove_change'] = (df ['Disapprove_rolling'] - df['Disapproving'])/df ['Disapprove_rolling'] *100
    df['Unsure_change'] = (df ['Unsure_rolling'] - df['Unsure/NoData'])/df ['Unsure_rolling'] *100
    return df


all_speechs = data_pull(all_speechs_data)

obama_approval = data_pull(obama_rating)
bush41_approval = data_pull(bush43_rating)
bush43_approval = data_pull(bush43_rating)
clinton_approval = data_pull(clinton_rating)
reagan_approval = data_pull(reagan_rating)
carter_approval = data_pull(carter_rating)
eisenhower_approval = data_pull(eisenhower_rating)
ford_approval = data_pull(ford_rating)
johnson_approval = data_pull(johnson_rating)
kennedy_approval = data_pull(kennedy_rating)
nixon_approval = data_pull(nixon_rating)
truman_approval = data_pull(truman_rating)
trump_approval = data_pull(trump_rating)

obama_approval = rolling_change(obama_approval)
bush41_approval = rolling_change(bush41_approval)
bush43_approval = rolling_change(bush43_approval)
clinton_approval = rolling_change(clinton_approval)
reagan_approval = rolling_change(reagan_approval)
carter_approval = rolling_change(carter_approval)
eisenhower_approval = rolling_change(eisenhower_approval)
ford_approval = rolling_change(ford_approval)
johnson_approval = rolling_change(johnson_approval)
kennedy_approval = rolling_change(kennedy_approval)
nixon_approval = rolling_change(nixon_approval)
truman_approval = rolling_change(truman_approval)
trump_approval = rolling_change(trump_approval)

all_speechs['Transcript'] = all_speechs['Transcript'].astype(str)
president_nsw = all_speechs[all_speechs.President.isin(president_list)]

obama_nsw = all_speechs[all_speechs["President"] == "Barack Obama"]

obama_nsw['Transcript']=obama_nsw['Transcript'].str.lower()
obama_nsw['Transcript']=obama_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#obama_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
obama_nsw['Transcript'] = obama_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in obama_nsw['Transcript']]
obama_nsw['Polarity'] = [b.sentiment.polarity for b in transcript_blob]
obama_nsw['Subjectivity'] = [b.sentiment.subjectivity for b in transcript_blob]
obama_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in obama_nsw['Transcript']]
obama_nsw['Negative'] = [analyzer.polarity_scores(v)['neg'] for v in obama_nsw['Transcript']]
obama_nsw['Neutral'] = [analyzer.polarity_scores(v)['neu'] for v in obama_nsw['Transcript']]
obama_nsw['Positive'] = [analyzer.polarity_scores(v)['pos'] for v in obama_nsw['Transcript']]


def sentiment_engine(president_name):
    df = all_speechs[all_speechs["President"] == president_name]

    df['Transcript']=df['Transcript'].str.lower()
    df['Transcript']=df['Transcript'].str.strip().str.replace('[^\w\s]','')
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    df['Transcript'] = df['Transcript'].str.replace(pat, '')
    transcript_blob = [TextBlob(desc) for desc in df['Transcript']]
    df['Polarity'] = [b.sentiment.polarity for b in transcript_blob]
    df['Subjectivity'] = [b.sentiment.subjectivity for b in transcript_blob]
    df['compound'] = [analyzer.polarity_scores(v)['compound'] for v in df['Transcript']]
    df['Negative'] = [analyzer.polarity_scores(v)['neg'] for v in df['Transcript']]
    df['Neutral'] = [analyzer.polarity_scores(v)['neu'] for v in df['Transcript']]
    df['Positive'] = [analyzer.polarity_scores(v)['pos'] for v in df['Transcript']]
    return df

def graphics (df,)

    labels = df['Date']
    neg = df['neg']
    neu = df['neu']
    pos = df['pos']

    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    ax.bar(labels, neg, width, label='Negative', color='red')
    ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
    ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
    ax.set_ylabel('Scores')
    ax.set_title('GW Bush Sentiment Scores by Date')
    ax.legend()
    plt.show()

labels = bush43_approval['Start Date']
neg = bush43_approval['Unsure/NoData']
neu = bush43_approval['Disapproving']
pos = bush43_approval['Approving']

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')

#plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
plt.xticks(rotation=45, ha='right', fontsize = 9)
ax.set_ylabel('Scores')
ax.set_title('GW Bush Approval Scores by Date')
ax.legend()
plt.show()

fig = plt.figure() 
plt.plot(bush_nsw.Date, bush_nsw.tb_Pol)
#fig.autofmt_xdate(rotation=45)
plt.xticks(rotation=45, ha='right')
plt.title('Bush44 Polarity')
plt.ylabel ('Polarity')
plt.show()


plt.plot(bush_nsw.Date, bush_nsw.tb_Subj)
plt.title('Bush44 Subjective')
plt.xlabel ('Date')
plt.ylabel ('Subjective')
plt.show()

plt.plot(bush_nsw.Date, bush_nsw.tb_Pol, label = 'Polarity')
plt.plot(bush_nsw.Date, bush_nsw.tb_Subj, label = 'Subjective')
plt.title('Bush combined')
plt.xlabel ('Date')
plt.ylabel ('Polarity')
plt.legend()
plt.show()

obama_sentiment = sentiment_engine("Barack Obama")
