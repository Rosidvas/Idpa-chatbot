import numpy as np
import pandas as pd

import re

from nltk.corpus import stopwords

limiter = 3000
df_ln = pd.read_csv("./datasets/sentences.csv")

df_en = df_ln[df_ln["lan_code"] == "eng"]
df_de = df_ln[df_ln["lan_code"] == "deu"]
df_fr = df_ln[df_ln["lan_code"] == "fra"]

df_en_x = df_en["sentence"].head(limiter)
df_de_x = df_de["sentence"].head(limiter)
df_fr_x = df_fr["sentence"].head(limiter)

df_en_y = df_en["lan_code"].head(limiter)
df_de_y = df_de["lan_code"].head(limiter)
df_fr_y = df_fr["lan_code"].head(limiter)

lx_index = [df_en_x, df_de_x, df_fr_x]
ly_index = [df_en_y, df_de_y, df_fr_y]

#convert raw target labels to lang codes
def restructure(y):
    lang = str(y)
    match lang:
        case "English":
            lang = "eng"
        case "German":
            lang = "deu"
        case "French":
            lang = "fra"
        case _:
            lang = lang
    return lang

#creates target and feature sets for language classification
def create_datasets():
    x_set = []
    y_set = [] 

    for lang in range(len(lx_index)):
        for example in range(len(lx_index[lang])):
            x_set.append(re.sub('[,.?!-]','', str(lx_index[lang].iloc[example])))
            y_set.append(restructure(ly_index[lang].iloc[example]))

    return x_set , y_set 

#function for testing dataset creation
def test():
    X_train, y_train = create_datasets()
    print(len(X_train))
    print(len(y_train))
    print(X_train[8550])
    print(y_train[8550])
    return
test()
