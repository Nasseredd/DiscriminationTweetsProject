#!/usr/bin/python3 
# -*- coding: utf-8 -*-

# Convert emoji 
def convert_emojis(text): 
    '''
        text : string 
    '''
    import emoji
    text = emoji.demojize(text)
    text = text.replace('_','')
    return text.replace(':','')


# Sub-dataframe 
def subdataframe(df, n): 
    '''
        Generate a DataFrame of n random examples from the original Dataframe  
        df : Original Dataframe from which a subdataframe is generated 
        n : number of examples (rows) 
    '''

    import random 
    i = random.randint(0,df.shape[0]-n)
    return df.iloc[i:i+n]

