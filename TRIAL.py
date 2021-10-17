import nltk
import nltk.corpus
from nltk.corpus import stopwords
#nltk.download("stopwords")
#nltk.download('punkt')
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

all_speechs = pd.read_csv('C:/Users/renje/Documents/Research Project/data/presidential_speeches.csv')
all_speechs['Transcript'] = all_speechs['Transcript'].astype(str)

president_list = ['Barack Obama', 'George W. Bush', 'Bill Clinton', 'George H. W. Bush']
president_nsw = all_speechs[all_speechs.President.isin(president_list)]

obama_nsw = all_speechs[(all_speechs["President"] == "Barack Obama"]

obama_nsw['Transcript']=obama_nsw['Transcript'].str.lower()
obama_nsw['Transcript']=obama_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#obama_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
obama_nsw['Transcript'] = obama_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in obama_nsw['Transcript']]
obama_nsw['tb_Pol'] = [b.sentiment.polarity for b in transcript_blob]
obama_nsw['tb_Subj'] = [b.sentiment.subjectivity for b in transcript_blob]
obama_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in obama_nsw['Transcript']]
obama_nsw['neg'] = [analyzer.polarity_scores(v)['neg'] for v in obama_nsw['Transcript']]
obama_nsw['neu'] = [analyzer.polarity_scores(v)['neu'] for v in obama_nsw['Transcript']]
obama_nsw['pos'] = [analyzer.polarity_scores(v)['pos'] for v in obama_nsw['Transcript']]


labels = obama_nsw['Date']
neg = obama_nsw['neg']
neu = obama_nsw['neu']
pos = obama_nsw['pos']

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg, label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive',color = 'green')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
ax.set_ylabel('Scores')
ax.set_title('Obama Scores by Date')
ax.legend()

plt.show()

plt.plot(obama_nsw.Date, obama_nsw.tb_Pol)
plt.title('Obama Polarity')
plt.xlabel ('Date')
plt.ylabel ('Polarity')
plt.show()

plt.plot(obama_nsw.Date, obama_nsw.tb_Subj)
plt.title('Obama Subjective')
plt.xlabel ('Date')
plt.ylabel ('Subjective')
plt.show()

plt.plot(obama_nsw.Date, obama_nsw.tb_Pol, label = 'Polarity')
plt.plot(obama_nsw.Date, obama_nsw.tb_Subj, label = 'Subjective')
plt.title('Obama combined')
plt.xlabel ('Date')
plt.ylabel ('Polarity')
plt.legend()
plt.show()


bush_nsw = all_speechs[all_speechs["President"] == "George W. Bush"]

bush_nsw['Transcript']=bush_nsw['Transcript'].str.lower()
bush_nsw['Transcript']=bush_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#bush_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
bush_nsw['Transcript'] = bush_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in bush_nsw['Transcript']]
bush_nsw['tb_Pol'] = [b.sentiment.polarity for b in transcript_blob]
bush_nsw['tb_Subj'] = [b.sentiment.subjectivity for b in transcript_blob]
bush_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in bush_nsw['Transcript']]
bush_nsw['neg'] = [analyzer.polarity_scores(v)['neg'] for v in bush_nsw['Transcript']]
bush_nsw['neu'] = [analyzer.polarity_scores(v)['neu'] for v in bush_nsw['Transcript']]
bush_nsw['pos'] = [analyzer.polarity_scores(v)['pos'] for v in bush_nsw['Transcript']]


labels = bush_nsw['Date']
neg = bush_nsw['neg']
neu = bush_nsw['neu']
pos = bush_nsw['pos']

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
ax.set_ylabel('Scores')
ax.set_title('GW Bush Scores by Date')
ax.legend()

plt.show()

plt.plot(bush_nsw.Date, bush_nsw.tb_Pol)
plt.title('Bush44 Polarity')
plt.xlabel ('Date')
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

clinton_nsw = all_speechs[all_speechs["President"] == "Bill Clinton"]

clinton_nsw['Transcript']=clinton_nsw['Transcript'].str.lower()
clinton_nsw['Transcript']=clinton_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#clinton_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
clinton_nsw['Transcript'] = clinton_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in clinton_nsw['Transcript']]
clinton_nsw['tb_Pol'] = [b.sentiment.polarity for b in transcript_blob]
clinton_nsw['tb_Subj'] = [b.sentiment.subjectivity for b in transcript_blob]
clinton_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in clinton_nsw['Transcript']]
clinton_nsw['neg'] = [analyzer.polarity_scores(v)['neg'] for v in clinton_nsw['Transcript']]
clinton_nsw['neu'] = [analyzer.polarity_scores(v)['neu'] for v in clinton_nsw['Transcript']]
clinton_nsw['pos'] = [analyzer.polarity_scores(v)['pos'] for v in clinton_nsw['Transcript']]


labels = clinton_nsw['Date']
neg = clinton_nsw['neg']
neu = clinton_nsw['neu']
pos = clinton_nsw['pos']

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
ax.set_ylabel('Scores')
ax.set_title('Clinton Scores by Date')
ax.legend()

plt.show()

hwbush_nsw = all_speechs[all_speechs["President"] == "George H. W. Bush"]

hwbush_nsw['Transcript']=hwbush_nsw['Transcript'].str.lower()
hwbush_nsw['Transcript']=hwbush_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#hwbush_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
hwbush_nsw['Transcript'] = hwbush_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in hwbush_nsw['Transcript']]
hwbush_nsw['tb_Pol'] = [b.sentiment.polarity for b in transcript_blob]
hwbush_nsw['tb_Subj'] = [b.sentiment.subjectivity for b in transcript_blob]
hwbush_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in hwbush_nsw['Transcript']]
hwbush_nsw['neg'] = [analyzer.polarity_scores(v)['neg'] for v in hwbush_nsw['Transcript']]
hwbush_nsw['neu'] = [analyzer.polarity_scores(v)['neu'] for v in hwbush_nsw['Transcript']]
hwbush_nsw['pos'] = [analyzer.polarity_scores(v)['pos'] for v in hwbush_nsw['Transcript']]


labels = hwbush_nsw['Date']
neg = hwbush_nsw['neg']
neu = hwbush_nsw['neu']
pos = hwbush_nsw['pos']

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


reagan_nsw = all_speechs[all_speechs["President"] == "Ronald Reagan"]

reagan_nsw['Transcript']=reagan_nsw['Transcript'].str.lower()
reagan_nsw['Transcript']=reagan_nsw['Transcript'].str.strip().str.replace('[^\w\s]','')
#hwbush_nsw['Transcript'].replace(stop_words,regex=True,inplace=True)
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
reagan_nsw['Transcript'] = reagan_nsw['Transcript'].str.replace(pat, '')

transcript_blob = [TextBlob(desc) for desc in reagan_nsw['Transcript']]
reagan_nsw['tb_Pol'] = [b.sentiment.polarity for b in transcript_blob]
reagan_nsw['tb_Subj'] = [b.sentiment.subjectivity for b in transcript_blob]
reagan_nsw['compound'] = [analyzer.polarity_scores(v)['compound'] for v in reagan_nsw['Transcript']]
reagan_nsw['neg'] = [analyzer.polarity_scores(v)['neg'] for v in reagan_nsw['Transcript']]
reagan_nsw['neu'] = [analyzer.polarity_scores(v)['neu'] for v in reagan_nsw['Transcript']]
reagan_nsw['pos'] = [analyzer.polarity_scores(v)['pos'] for v in reagan_nsw['Transcript']]


labels = reagan_nsw['Date']
neg = reagan_nsw['neg']
neu = reagan_nsw['neu']
pos = reagan_nsw['pos']

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, neg, width, label='Negative', color='red')
ax.bar(labels, neu, width, bottom=neg,label='Neutral', color ='gray')
ax.bar(labels, pos, width, bottom=neg+neu, label='Postive', color = 'green')

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize = 9) 
ax.set_ylabel('Scores')
ax.set_title('Reagan Scores by Date')
ax.legend()

plt.show()
