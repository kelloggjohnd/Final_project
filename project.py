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
speech_numbers = "https://github.com/kelloggjohnd/Final_project/blob/main/data/presidential_speeches_numbers.csv"

def data_pull(url):
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return df

def rolling_change(df,President):
    df['President'] = President
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
    df = df[['Date','Polarity','Subjectivity','compound','Negative','Neutral','Positive', 'Approve_change','Disapprove_change','Unsure_change','week_year']]
    
    return df

def long_form (merge_df, column_value):
    df = merge_df[[column_value,'Approve_change', 'Disapprove_change','Unsure_change']]
    df = pd.melt(df, id_vars=column_value, value_vars=['Approve_change','Disapprove_change','Unsure_change'])
    
def long_form_adjust (parentdf, President, column_value):    
    df = parentdf[parentdf['President'] == President]
    df = df[[column_value,'Approve_change', 'Disapprove_change','Unsure_change']]
    df = pd.melt(df, id_vars=column_value, value_vars=['Approve_change','Disapprove_change','Unsure_change'])
    
    return df

all_speechs = data_pull(all_speechs_data)
#speech_poll_numbers = data_pull(speech_numbers)
speech_poll_numbers = pd.read_csv (r'C:\Users\renje\Documents\GitHub\Final_project\data\presidential_speeches_numbers.csv')

obama_approval = data_pull(obama_rating)
bush41_approval = data_pull(bush41_rating)
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


obama_approval = rolling_change(obama_approval, "Barack Obama")
bush41_approval = rolling_change(bush41_approval, "George H.W. Bush")
bush43_approval = rolling_change(bush43_approval, 'George W. Bush')
clinton_approval = rolling_change(clinton_approval, 'Bill Clinton')
reagan_approval = rolling_change(reagan_approval, 'Ronald Reagan')
carter_approval = rolling_change(carter_approval, 'Jimmy Carter')
eisenhower_approval = rolling_change(eisenhower_approval, 'Dwight D. Eisenhower')
ford_approval = rolling_change(ford_approval, "Gerald Ford")
johnson_approval = rolling_change(johnson_approval, 'Lyndon B. Johnson')
kennedy_approval = rolling_change(kennedy_approval, 'John F. Kennedy')
nixon_approval = rolling_change(nixon_approval, 'Richard Nixon')
truman_approval = rolling_change(truman_approval,'Herry S. Truman')


gallup_totals = pd.concat([obama_approval,bush43_approval])
gallup_totals = pd.concat([gallup_totals,clinton_approval])
gallup_totals = pd.concat([gallup_totals,bush41_approval])
gallup_totals = pd.concat([gallup_totals,reagan_approval])
gallup_totals = pd.concat([gallup_totals,carter_approval])
gallup_totals = pd.concat([gallup_totals,ford_approval])
gallup_totals = pd.concat([gallup_totals,nixon_approval])
gallup_totals = pd.concat([gallup_totals,johnson_approval])
gallup_totals = pd.concat([gallup_totals,kennedy_approval])
gallup_totals = pd.concat([gallup_totals,eisenhower_approval])
gallup_totals = pd.concat([gallup_totals,truman_approval])

gallup_totals = gallup_totals[['Start Date','President','Approving', 'Disapproving', 'Unsure/NoData']]
gallup_totals = pd.melt(gallup_totals, id_vars=['Start Date','President'], value_vars=['Approving', 'Disapproving', 'Unsure/NoData'])

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

truman_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\truman.csv', sep=',')
eisenhower_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\eisenhower.csv', sep=',')
kennedy_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\kennedy.csv', sep=',')
johnson_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\johnson.csv', sep=',')
nixon_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\nixon.csv', sep=',')
ford_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\ford.csv', sep=',')
carter_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\carter.csv', sep=',')
reagan_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\reagan.csv', sep=',')
bush41_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\bush41.csv', sep=',')
clinton_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\clinton.csv', sep=',')
bush43_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\bush43.csv', sep=',')
obama_approval.to_csv(r'C:\Users\renje\Documents\GitHub\Final_project\data\approval_ratings\Adjusted\obama.csv', sep=',')

truman_merge = truman_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
eisenhower_merge = eisenhower_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
kennedy_merge = kennedy_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
johnson_merge = johnson_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
nixon_merge = nixon_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
ford_merge = ford_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
carter_merge = carter_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
reagan_merge = reagan_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
bush41_merge = bush41_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
clinton_merge = clinton_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
bush43_merge = bush43_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]
obama_merge = obama_speeches[['Date', 'President', 'Polarity', 'Subjectivity','compound','Negative', 'Neutral','Positive']]

All_merge = pd.concat([obama_merge,bush43_merge])
All_merge = pd.concat([All_merge,clinton_merge])
All_merge = pd.concat([All_merge,bush41_merge])
All_merge = pd.concat([All_merge,reagan_merge])
All_merge = pd.concat([All_merge,carter_merge])
All_merge = pd.concat([All_merge,ford_merge])
All_merge = pd.concat([All_merge,nixon_merge])
All_merge = pd.concat([All_merge,johnson_merge])
All_merge = pd.concat([All_merge,kennedy_merge])
All_merge = pd.concat([All_merge,eisenhower_merge])
All_merge = pd.concat([All_merge,truman_merge])

All_merge['Date'] = pd.to_datetime(All_merge['Date'])
speech_poll_numbers['Date'] = pd.to_datetime(speech_poll_numbers['Date'])

merged = speech_poll_numbers.merge(All_merge, left_on=['Date','President'], right_on = ['Date','President'])
merged = merged.drop(columns=['Approve_rolling','Disapprove_rolling','Unsure_rolling'])

All_merge_numbers = All_merge.groupby('President').mean()

gallup_totals = pd.concat([obama_approval,bush43_merge])
All_merge = pd.concat([All_merge,clinton_merge])
All_merge = pd.concat([All_merge,bush41_merge])
All_merge = pd.concat([All_merge,reagan_merge])
All_merge = pd.concat([All_merge,carter_merge])
All_merge = pd.concat([All_merge,ford_merge])
All_merge = pd.concat([All_merge,nixon_merge])
All_merge = pd.concat([All_merge,johnson_merge])
All_merge = pd.concat([All_merge,kennedy_merge])
All_merge = pd.concat([All_merge,eisenhower_merge])
All_merge = pd.concat([All_merge,truman_merge])

###########Graphics###########

pio.templates.default = "simple_white"

def speech_graphs (df, President, merge_df, approval_df):
    
    df_polarity = long_form_adjust(df,President,'Polarity')
    fig1 = px.scatter(df_polarity, x="Polarity", y="value", facet_row="variable", trendline='ols', title= President+" Polarity")
    
    df_subjectivity = long_form_adjust(df,President,'Subjectivity')
    fig2 = px.scatter(df_subjectivity, x="Subjectivity", y="value", facet_row="variable", trendline='ols', title=President+" Subjectivity")
        
    df_positive = long_form_adjust(df,President,'Positive')
    fig3 = px.scatter(df_positive, x="Positive", y="value", facet_row="variable", trendline='ols', title=President+" Positive")
    
    df_negative = long_form_adjust(df,President,'Negative')
    fig4 = px.scatter(df_negative, x="Negative", y="value", facet_row="variable", trendline='ols', title=President+" Negative")
    
    df_neutral = long_form_adjust(df,President,'Neutral')
    fig5 = px.scatter(df_neutral, x="Neutral", y="value", facet_row="variable", trendline='ols', title=President+" Neutral")
    
    df_numbers = merge_df[['Date','Negative','Neutral','Positive']]
    df_numbers = pd.melt(df_numbers, id_vars='Date', value_vars=['Negative','Neutral','Positive'])
    
    fig6 = px.bar(df_numbers, x="Date", y="value", color="variable", title=President+" Speech Breakdown")
    fig6.update_xaxes(type='category')
        
    df_polar_subj = merge_df[['Date','Polarity','Subjectivity']]
    df_polar_subj = pd.melt(df_polar_subj, id_vars='Date', value_vars=['Polarity','Subjectivity'])
    
    fig7 = px.line(df_polar_subj, x = 'Date', y='value', color='variable', facet_row='variable', title = President+" Polarity & Subjectivity")
    fig7.update_xaxes(type='category')
    fig7.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    
    approval = approval_df [['Start Date','Approving', 'Disapproving', 'Unsure/NoData']]
    approval = pd.melt(approval, id_vars='Start Date', value_vars=['Approving', 'Disapproving', 'Unsure/NoData'])
    
    fig8 = px.line(approval, x="Start Date", y="value", color="variable", facet_row='variable',title=President+" Approval Breakdown")
    
    
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()
    fig6.show()
    fig7.show()
    fig8.show()
    
    return fig1, fig2, fig3, fig4,fig5,fig6,fig7,fig8

###########Gallup###########
gallup_totals['Start Date'] = pd.to_datetime(gallup_totals['Start Date'])
gallup_totals['Start Date']= gallup_totals['Start Date'].dt.date
gallup_totals.sort_values('Start Date', inplace=True)

gallup_fig = px.line(gallup_totals, x="Start Date", y="value", color="variable", facet_row='variable',title="Gallup Breakdown")
gallup_fig.update_xaxes(type='category')
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
gallup_fig.show()

############CORR Plots##############
import matplotlib
matplotlib.use("TKAgg")

corr = merged.corr()

############Obama##############
speech_graphs(merged,"Barack Obama", obama_merge,obama_approval)

############Bush43##############
speech_graphs(merged,"George W. Bush", bush43_merge,bush43_approval)

############clinton##############
speech_graphs(merged,"Bill Clinton", clinton_merge,clinton_approval)

############Bush41##############
speech_graphs(merged,"George H. W. Bush", bush41_merge,bush41_approval)

############reagan##############
speech_graphs(merged,"Ronald Reagan", reagan_merge,reagan_approval)

############Carter##############
speech_graphs(merged,"Jimmy Carter", carter_merge,carter_approval)

############ford##############
speech_graphs(merged,"Gerald Ford", ford_merge,ford_approval)

############Nixon##############
speech_graphs(merged,"Richard M. Nixon", nixon_merge,nixon_approval)

############Johnson##############
speech_graphs(merged,"Lyndon B. Johnson", johnson_merge,johnson_approval)

############Kennedy##############
speech_graphs(merged,"John F. Kennedy", kennedy_merge,kennedy_approval)

############Eisenhower##############
speech_graphs(merged,"Dwight D. Eisenhower", eisenhower_merge,eisenhower_approval)

############Truman##############
speech_graphs(merged,"Harry S. Truman", truman_merge,truman_approval)




approval_test = obama_approval [['Start Date','Approving']]
    
fig8 = px.line(approval_test, x="Start Date", y="Approving",title="Obama Approval")
fig8.update_layout(yaxis={"dtick":5,"range":[0,100]})
fig8.show()

fig30 = go.Figure()
