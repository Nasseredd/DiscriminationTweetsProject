#!/usr/bin/python3 
# -*- coding: utf-8 -*-

import pandas as pd 

def results(*args):
    '''
        [Method,Accuracy,F1Score]
    '''
    
    methods_results = []
    for method in args:
        methods_results.append(method)
    return pd.DataFrame(methods_results,columns=["Methods","Accuracy","F1 Score"]) 

def dependency_graph(tweets):
    import spacy 
    nlp = spacy.load('en_core_web_sm')
    tweet_dependencies = []
    
    for tweet in tweets:
        doc = nlp(tweet)
        all_two_treelet = ""
        for token in doc:
            two_treelet = str(token.pos_) + " -> " + str(token.dep_) + " -> " + str(token.head.pos_)
            all_two_treelet += two_treelet + " | "
        tweet_dependencies.append(all_two_treelet)

    return pd.Series(tweet_dependencies)

def ngrams_frequency(tweets, grams):
    
    # all tweets in one single string 
    all_tweets = tweets.sum(axis=0)
    
    ngrams_frequency = []
    for gram in grams:
        ngrams_frequency.append([gram,all_tweets.count(gram)])
    
    matrix = pd.DataFrame(ngrams_frequency,columns=['Gram','Frequencies'])
    matrix.sort_values(by='Frequencies', ascending=False, inplace=True)
    return matrix

def ngram(tweets, nbr_gram, stopwords):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=None, 
                                stop_words=stopwords,
                                ngram_range = (nbr_gram,nbr_gram)
                                )
    vectorizer.fit(tweets)
    vocab = list(vectorizer.vocabulary_.keys())
    
    return vocab 


def named_entity_recognition(tweets):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    bloc = []
    for tweet in tweets: 
        doc = nlp(tweet)
        for ent in doc.ents:
            bloc.append([ent.text, ent.label_])
    return pd.DataFrame(bloc,columns=['Text','Label'])


def len_tweets(tweets):
    length = []
    for tweet in tweets:
        length.append(len(tweet))
    return pd.DataFrame(length)

def subjectivity(tweets):
    from textblob import TextBlob
    subjectivity = []
    for tweet in tweets: 
        pol = TextBlob(tweet)
        subjectivity.append(pol.sentiment.subjectivity)
    return pd.DataFrame(subjectivity)

def polarity(tweets):
    from textblob import TextBlob
    polarity = []
    for tweet in tweets: 
        pol = TextBlob(tweet)
        polarity.append(pol.sentiment.polarity)
    return pd.DataFrame(polarity)


def combine_features(X, *new_features):
    X = pd.DataFrame(X)
    F = pd.concat(list(new_features), axis=1)
    return pd.concat([F,X], axis=1)


def newfeatures(newf, tweets):
    matrix = []
    for tweet in tweets:
        values = []
        for feature in newf:
            if feature in tweet: values.append(1)
            else: values.append(0)
        matrix.append(values)
    return pd.DataFrame(matrix, columns=newf)