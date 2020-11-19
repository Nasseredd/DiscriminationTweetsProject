#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



def canon_extract(df, nbr_tokens):
    from DPreprocessing import preprocessing
    from Models import support_vector_machine

    # Outputs 
    metrics = []
    f1_scores = []
    accuracy_scores = []

    # Temporary lists 
    f_stem = []
    a_stem = []
    f_lem = []
    a_lem = []

    for nbr in nbr_tokens:
        # Stemming
        X_train, X_test, y_train, y_test = preprocessing(dataset=df, nbr_tokens=nbr,  
                                                         vectorizer="tfidf", canon="stemming")
        f_stem.append(support_vector_machine(X_train, X_test, y_train, y_test,"f1score"))
        a_stem.append(support_vector_machine(X_train, X_test, y_train, y_test,"accuracy"))

        # Lemmatization
        X_train, X_test, y_train, y_test = preprocessing(dataset=df, nbr_tokens=nbr,  
                                                         vectorizer="tfidf", canon="lemmatization")
        f_lem.append(support_vector_machine(X_train, X_test, y_train, y_test,"f1score"))
        a_lem.append(support_vector_machine(X_train, X_test, y_train, y_test,"accuracy"))

    f1_scores = [f_stem,f_lem]
    accuracy_scores = [a_stem,a_lem]

    final_f1score_stemming = f1_scores[0][-1]
    final_f1score_lemmatization = f1_scores[1][-1]
    final_accuracy_stemming = accuracy_scores[0][-1]
    final_accuracy_lemmatization = accuracy_scores[1][-1]


    metrics = [final_f1score_stemming,final_f1score_lemmatization,
               final_accuracy_stemming,final_accuracy_lemmatization]    

    return metrics, f1_scores, accuracy_scores

def metrics_graphic(stemming_metrics, lemmatization_metrics):
    '''
        stemming_metrics = []
        lemmatization_metrics = []
    '''

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7,7))
    N = 2
    ind = np.arange(N) 
    width = 0.2       
    plt.bar(ind, stemming_metrics, width, label='Stemming', color='b', edgecolor='black')
    plt.bar(ind + width, lemmatization_metrics, width, label='Lemmatization', color='m', edgecolor='black')
    plt.xticks(ind + width / 2, ('Accuracy', 'F1 Score'))
    plt.legend(loc='best')
    plt.show()

def evolution_f1score_graphic(f1_lists, nbr_tokens):
    import matplotlib.pyplot as plt
    #_, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True,figsize=(16, 6))
    
    plt.figure(figsize=(7,7))
    curves = []
    labels = ['Stemming','Lemmatization & POS']
    
    for liste, l in zip(f1_lists, labels):
        curves.extend(plt.plot(nbr_tokens, liste, '-p', label=l))
    
    #plt.hlines(y = 0.7, xmin = 120, xmax = 3050, color ='r')
    #plt.text(1, 0.7, '70%', ha ='left', va ='center')
    
    plt.legend(handles=curves)
    plt.xlabel('Number of most frequent tokens')
    plt.ylabel('F1-Score')
    plt.ylim(-0.25,1)
    plt.show()

def evolution_accuracy_graphic(accuracy_lists, nbr_tokens):
    import matplotlib.pyplot as plt

    curves = []
    labels = ['Stemming','Lemmatization & POS']
    
    for liste, l in zip(accuracy_lists, labels):
        curves.extend(plt.plot(nbr_tokens, liste, '-p', label=l))
    
    #plt.hlines(y = 0.7, xmin = 120, xmax = 3050, color ='r')
    #plt.text(1, 0.7, '70%', ha ='left', va ='center')
    
    plt.legend(handles=curves)
    plt.xlabel('Number of most frequent tokens')
    plt.ylabel('Accuracy')
    plt.ylim(-0.25,1)

    plt.show()

def canonization_comparison_graphic(metrics, f1_lists, accuracy_lists, nbr_tokens):
    ''' metrics = ['Stemming accuracy','Stemming f1 score','Lemmatization accuracy','Lemmatization f1 score']'''
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # figure 1 
    stemming_metrics = [metrics[0], metrics[1]]
    lemmatization_metrics = [metrics[2], metrics[3]]
    metrics_graphic(stemming_metrics, lemmatization_metrics)
    
    # figure 2 
    evolution_accuracy_graphic(accuracy_lists, nbr_tokens)
    
    
    # figure 3 
    evolution_accuracy_graphic(accuracy_lists, nbr_tokens)
    
    #plt.subplots_adjust(wspace = 0.5)
    plt.show()

def canonization_comparison(df, nbr_tokens):
    metrics, f1_lists, accuracy_lists = canon_extract(df, nbr_tokens)
    canonization_comparison_graphic(metrics, f1_lists, accuracy_lists, nbr_tokens) 
