from PIL._imaging import display
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

def percentage(part, whole):
    return 100 * float(part) / float(whole)


def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


# Removing Punctuation
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0–9]+', '', text)
    return text


# Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


def stemming(text):
    text = [ps.stem(word) for word in text]
    return text


# Cleaning Text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)  # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


#Function to ngram
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# Authentication
consumerKey = "5aORitGDxbOc7loiZxFLK4i0E"
consumerSecret = "aeUIWYlbTCnSybmzWWXa884onRx8FU0gQA2ZqxGnfsEfklEIVO"
accessToken = "1392798908406018050-Jds5ZnYVaRgFLBki61HoKdZWjMfrgU"
accessTokenSecret = "SBRXURhvPRo5KC2lAHBK9sUutaB6zzkFv9rEHoA93xSeT"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)


###########1. Data Collection###############

#keyword = input("Please enter keyword or hashtag to search: ")
keyword = "Microsoft"
#noOfTweet = int(input("Please enter how many tweets to analyze: "))
noOfTweet = 20
tweets = tweepy.Cursor(api.search, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

for tweet in tweets:
    print(tweet.text)
    tweet_list.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity = polarity + analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(str(tweet.text))
        negative = negative + 1

    elif pos > neg:
        positive_list.append(str(tweet.text))
        print(positive)
        positive = positive + 1

    elif pos == neg:
        neutral_list.append(str(tweet.text))
        neutral = neutral + 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')

#Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print('\ntotal number: ',len(tweet_list))
print('positive number: ',len(positive_list))
print('negative number: ', len(negative_list))
print('neutral number: ',len(neutral_list))

#Creating PieCart
labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= "+keyword)
plt.axis("equal")
plt.show()

#Cleaning Text (RT, Punctuation etc)
#Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]
print("\n*******TW list********")
print(tw_list)
#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\\w+: ',' ',x)
rt = lambda x: re.sub('[^A-Za-z0-9]+', ' ',x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
#tw_list.head(10)
print("\n*******Edited TW list********")
print(tw_list)

#Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
       tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
       tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp


pd.set_option('display.max_columns', None)
#tw_list.head(10)
print("\n*******Calculating Negative, Positive, Neutral and Compound values********")
print(tw_list)

#Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]

#Count_values for sentiment
result = count_values_in_column(tw_list,"sentiment")
print("\n")
print(result)

#Create data for Pie Chart
pichart = count_values_in_column(tw_list,"sentiment")
names = result.index
size = result["Percentage"]

#Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

#Calculating tweet’s lenght and word count
tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))
print("\n")
print(round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2))

print("\n")
print(round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2))

###########2. Data Pre Processing###############

tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))

tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))
# Removing stopwords
stopword = nltk.corpus.stopwords.words('english')

tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))
# Appliyng Stemmer
ps = nltk.PorterStemmer()

tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

#tw_list.head()
print("\n*******Punct, tokenized, nonstop, stemmed********")
print(tw_list)

#Apply count vectorizer the see all unique words as a new features
#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
print("\n")
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())
count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
#count_vect_df.head()
print(count_vect_df)

# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
print("\n")
print(countdf[1:11])

#n2_bigram
n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)
print("\nBi grams")
print(n2_bigrams)

#n3_trigram
n3_trigrams = get_top_n_gram(tw_list['text'],(3,3),20)
print("\nTri grams")
print(n3_trigrams)

print("\ncountVectorizer.get_feature_names():")
print(countVectorizer.get_feature_names())

#Word2Vec
word2vec = Word2Vec(countVectorizer.get_feature_names(), min_count=2)
vocabulary = list(word2vec.wv.index_to_key)
print("\nWord2Vec")
print(vocabulary)

#for index, word in enumerate(word2vec.wv.index_to_key):
#    print(f"word #{index}/{len(word2vec.wv.index_to_key)} is {word}")

for t in tw_list["text"]:
    #print(t)
    word2vec = Word2Vec(t, min_count=2)
    vocabulary = list(word2vec.wv.index_to_key)
    print(vocabulary)