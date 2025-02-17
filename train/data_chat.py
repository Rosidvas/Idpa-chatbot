import json
import nltk
import pandas as pd
import numpy as np


with open("./datasets/reference_en_v1.json", "r") as file:
    data_en = json.load(file)

with open("./datasets/reference_de_v1.json", "r") as file:
    data_de = json.load(file)

with open("./datasets/reference_fr_v1.json", "r") as file:
    data_fr = json.load(file)

def extract_data():
    return data_de, data_en, data_fr

def test():
    deutsch, briish, franz = extract_data()
    print(franz)

#test()