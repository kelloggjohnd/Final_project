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
from plotly.subplots import make_subplots
pio.renderers.default='browser'


import pandas as pd
import numpy as np
import datetime as dt
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

def week_number_approve (df):
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['week_number'] = df['End Date'].dt.week
    df['year'] = df['End Date'].dt.year
    df['week_year'] = df['week_number'].map(str)+'-'+df['year'].map(str)
    
    return df

def week_number_speeches (df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['week_number'] = df['Date'].dt.week
    df['year'] = df['Date'].dt.year
    df['week_year'] = df['week_number'].map(str)+'-'+df['year'].map(str)
    
    return df

def data_merge (speeches, approval):
    df = speeches.merge(approval,how='inner', left_on='week_year', right_on='week_year')
    df = df[['Date','Polarity','Subjectivity','compound','Negative','Neutral','Positive', 'Approve_change','Disapprove_change','Unsure_change']]
    
    return df

def long_form (merge_df, column_value):
    df = merge_df[[column_value, 'Approve_change', 'Disapprove_change', 'Unsure_change']]
    df = pd.melt(df, id_vars=column_value, value_vars=['Approve_change','Disapprove_change','Unsure_change'])
    
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

all_speechs['Transcript'] = all_speechs['Transcript'].astype(str)
president_speeches = all_speechs[all_speechs.President.isin(president_list)]

truman_speeches = sentiment_engine('Harry S. Truman')
truman_speeches = truman_speeches[~(truman_speeches['Date'] < '1945-04-12')]
truman_speeches = truman_speeches[~(truman_speeches['Date'] > '1953-01-20')]

eisenhower_speeches = sentiment_engine('Dwight D. Eisenhower')
eisenhower_speeches = eisenhower_speeches[~(eisenhower_speeches['Date'] < '1953-01-20')]
eisenhower_speeches = eisenhower_speeches[~(eisenhower_speeches['Date'] > '1961-01-20')]

kennedy_speeches = sentiment_engine('John F. Kennedy')
kennedy_speeches = kennedy_speeches[~(kennedy_speeches['Date'] < '1961-01-20')]
kennedy_speeches = kennedy_speeches[~(kennedy_speeches['Date'] > '1963-11-22')]

johnson_speeches = sentiment_engine('Lyndon B. Johnson')
johnson_speeches = johnson_speeches[~(johnson_speeches['Date'] < '1963-11-22')]
johnson_speeches = johnson_speeches[~(johnson_speeches['Date'] > '1969-01-20')]

nixon_speeches = sentiment_engine('Richard M. Nixon')
nixon_speeches = nixon_speeches[~(nixon_speeches['Date'] < '1969-01-20')]
nixon_speeches = nixon_speeches[~(nixon_speeches['Date'] > '1974-08-09')]

ford_speeches = sentiment_engine('Gerald Ford')
ford_speeches = ford_speeches[~(ford_speeches['Date'] < '1974-08-09')]
ford_speeches = ford_speeches[~(ford_speeches['Date'] > '1977-01-20')]

carter_speeches = sentiment_engine('Jimmy Carter')
carter_speeches = carter_speeches[~(carter_speeches['Date'] < '1977-01-20')]
carter_speeches = carter_speeches[~(carter_speeches['Date'] > '1981-01-20')]

reagan_speeches = sentiment_engine('Ronald Reagan')
reagan_speeches = reagan_speeches[~(reagan_speeches['Date'] < '1981-01-20')]
reagan_speeches = reagan_speeches[~(reagan_speeches['Date'] > '1989-01-20')]

bush41_speeches = sentiment_engine('George H. W. Bush')
bush41_speeches = bush41_speeches[~(bush41_speeches['Date'] < '1989-01-20')]
bush41_speeches = bush41_speeches[~(bush41_speeches['Date'] > '1993-01-20')]

clinton_speeches = sentiment_engine('Bill Clinton')
clinton_speeches = clinton_speeches[~(clinton_speeches['Date'] < '1993-01-20')]
clinton_speeches = clinton_speeches[~(clinton_speeches['Date'] > '2001-01-20')]

bush43_speeches = sentiment_engine('George W. Bush')
bush43_speeches = bush43_speeches[~(bush43_speeches['Date'] < '2001-01-20')]
bush43_speeches = bush43_speeches[~(bush43_speeches['Date'] > '2009-01-20')]

obama_speeches = sentiment_engine('Barack Obama')
obama_speeches = obama_speeches[~(obama_speeches['Date'] < '2009-01-20')]
obama_speeches = obama_speeches[~(obama_speeches['Date'] > '2017-01-20')]


obama_approval = week_number_approve(obama_approval)
obama_speeches = week_number_speeches(obama_speeches)
obama_merge = data_merge(obama_speeches,obama_approval)

bush43_approval = week_number_approve(bush43_approval)
bush43_speeches = week_number_speeches(bush43_speeches)
bush43_merge = data_merge(bush43_speeches,bush43_approval)

clinton_approval = week_number_approve(clinton_approval)
clinton_speeches = week_number_speeches(clinton_speeches)
bush43_merge = data_merge(clinton_speeches,clinton_approval)

bush41_approval = week_number_approve(bush41_approval)
bush41_speeches = week_number_speeches(bush41_speeches)
bush41_merge = data_merge(bush41_speeches,bush41_approval)

reagan_approval = week_number_approve(reagan_approval)
reagan_speeches = week_number_speeches(reagan_speeches)
reagan_merge = data_merge(reagan_speeches,reagan_approval)

carter_approval = week_number_approve(carter_approval)
carter_speeches = week_number_speeches(carter_speeches)
carter_merge = data_merge(carter_speeches,carter_approval)

ford_approval = week_number_approve(ford_approval)
ford_speeches = week_number_speeches(ford_speeches)
ford_merge = data_merge(ford_speeches,ford_approval)

nixon_approval = week_number_approve(nixon_approval)
nixon_speeches = week_number_speeches(nixon_speeches)
nixon_merge = data_merge(nixon_speeches,nixon_approval)

johnson_approval = week_number_approve(johnson_approval)
johnson_speeches = week_number_speeches(johnson_speeches)
johnson_merge = data_merge(johnson_speeches,johnson_approval)

kennedy_approval = week_number_approve(kennedy_approval)
kennedy_speeches = week_number_speeches(kennedy_speeches)
kennedy_merge = data_merge(kennedy_speeches,kennedy_approval)

eisenhower_approval = week_number_approve(eisenhower_approval)
eisenhower_speeches = week_number_speeches(eisenhower_speeches)
eisenhower_merge = data_merge(eisenhower_speeches,eisenhower_approval)

truman_approval = week_number_approve(truman_approval)
truman_speeches = week_number_speeches(truman_speeches)
truman_merge = data_merge(truman_speeches,truman_approval)

###########Graphics###########

############Obama##############
obama_polarity = long_form(obama_merge, 'Polarity')
fig1 = px.scatter(obama_polarity, x="Polarity", y="value", facet_row="variable", trendline='ols', title="Obama Polarity")
fig1.show()

obama_subjectivity = long_form(obama_merge, 'Subjectivity')
fig2 = px.scatter(obama_subjectivity, x="Subjectivity", y="value", facet_row="variable", trendline='ols', title="Obama Subjectivity")
fig2.show()

obama_positive = long_form(obama_merge, 'Positive')
fig3 = px.scatter(obama_positive, x="Positive", y="value", facet_row="variable", trendline='ols', title="Obama Positive")
fig3.show()

obama_negative = long_form(obama_merge, 'Negative')
fig4 = px.scatter(obama_negative, x="Negative", y="value", facet_row="variable", trendline='ols', title="Obama Negative")
fig4.show()

obama_neutral = long_form(obama_merge, 'Neutral')
fig5 = px.scatter(obama_neutral, x="Neutral", y="value", facet_row="variable", trendline='ols', title="Obama Neutral")
fig5.show()

obama_numbers = obama_speeches[['Date','Negative','Neutral','Positive']]
obama_numbers = pd.melt(obama_numbers, id_vars='Date', value_vars=['Negative','Neutral','Positive'])


fig = px.bar(obama_numbers, x="Date", y="value", color="variable", title="Obama Speech Breakdown")
fig.update_xaxes(type='category')
fig.show()

obama_polar_subj = obama_speeches[['Date','Polarity','Subjectivity']]
obama_polar_subj = pd.melt(obama_polar_subj, id_vars='Date', value_vars=['Polarity','Subjectivity'])

fig = px.line(obama_polar_subj, x = 'Date', y='value', color='variable', facet_row='variable' )
fig.update_xaxes(type='category')
fig.show()




####################
bush43_polarity = long_form(bush43_merge, 'Polarity')
fig6 = px.scatter(bush43_polarity, x="Polarity", y="value", facet_row="variable", trendline='ols', title="George W Bush Polarity")
fig6.show()

bush43_subjectivity = long_form(bush43_merge, 'Subjectivity')
fig7 = px.scatter(bush43_subjectivity, x="Subjectivity", y="value", facet_row="variable", trendline='ols', title="George W Bush Subjectivity")
fig7.show()

bush43_positive = long_form(bush43_merge, 'Positive')
fig8 = px.scatter(bush43_positive, x="Positive", y="value", facet_row="variable", trendline='ols', title="George W Bush Positive")
fig8.show()

bush43_negative = long_form(bush43_merge, 'Negative')
fig9 = px.scatter(bush43_negative, x="Negative", y="value", facet_row="variable", trendline='ols', title="George W Bush Negative")
fig9.show()

bush43_neutral = long_form(bush43_merge, 'Neutral')
fig10 = px.scatter(bush43_neutral, x="Neutral", y="value", facet_row="variable", trendline='ols', title="George W Bush Neutral")
fig10.show()


labels = bush43_speeches['Date']
neg = bush43_speeches['Negative']
neu = bush43_speeches['Neutral']
pos = bush43_speeches['Positive']

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
ax.set_ylabel('Scores')
ax.set_title('GHW Bush Scores by Date')
ax.legend()

plt.show()
