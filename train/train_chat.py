import nltk
import random
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, GermanStemmer, FrenchStemmer
from data_chat import extract_data


stopwords_en = stopwords.words('english')
stopwords_de = stopwords.words('german')
stopwords_fr = stopwords.words('french')

classes = []
words = []
intents = []
ignore = ["?",".","!"]
documents = []

#Language selection for stemmerizer
def stemmerizer(language):
    match language:
        case "english":
            return EnglishStemmer()
        case "german":
            return GermanStemmer()
        case "french":
            return FrenchStemmer()
        case _:
            return EnglishStemmer()
        
#Restructure dataset and extract intents
def extract_intents(data, language): 
    global words
    global intents
    global classes

    stemmer = stemmerizer(language=language)
    intents = data["intents"]
    for intent in intents:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern, language=language)
            words.extend(w)

            documents.append((w, intent["tag"]))

            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
    words = sorted(list(set(words)))
        
#Converts patterns to inputs for Tensorflow
def vectorization(language):
    global words
    global intents
    global classes
    training = []
    output = []
    output_empty = [0] * len(classes)
    stemmer = stemmerizer(language=language)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        output_layer = list(output_empty)
        try: 
            output_layer[classes.index(doc[1])] = 1
        except IndexError:
            print(f"Index out of range. class length: {len(classes)} \n output_layer len: {len(output_layer)}")
        training.append(bag)
        output.append(output_layer)

    training = np.array(training)
    output = np.array(output)
    print(f"train shape 1: {training.shape}")
    training = np.array(training)
    output = np.array(output)
    print(f"train shape 2: {training.shape}")
    #print(len(output_layer))

    return training, output

#Builds, trains and saves Neural Network model
def build_model(training, output):      
    input_layer = tf.keras.layers.Input(shape=(len(training[0]),))
    layer1 = tf.keras.layers.Dense(32, activation='relu', name='L1')(input_layer)
    layer2 = tf.keras.layers.Dense(32, activation='relu', name='L2')(layer1)
    output_layer = tf.keras.layers.Dense(len(output[0]), activation='linear')(layer2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.summary()

    #Starts model training then shows graph of the model training result
    history = model.fit(training, output, epochs=50, batch_size=16, verbose=1)
    print("training ended")
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()
    
    #Change save model, dialogue names when switching languages to train either --> de/fr/en
    model_name = "m_de_chatbot_v1.keras"
    model.save(f"./models/chatbot/{model_name}")
    
    dialogue_name = "de_dialogue_v1.pkl" 
    with open(f"./models/chatbot/{dialogue_name}", "wb") as file:
        data_to_save = {"words": words, "classes": classes}
        pickle.dump(data_to_save, file)
    
    return model

"""
-----------------------------------------------------------
Functions made for testing currently trained Neural Network
-----------------------------------------------------------
"""

#get random response from the dialogue json
def get_response(intent_index):
    global intents
    recognized_intent = classes[intent_index]
    print(recognized_intent)
    for intent in intents:
        if intent["tag"] == recognized_intent:
            response = random.choice(intent["responses"])
            break

    return recognized_intent

#test the intent classification model: only sends intents as response.
def test_chat(model, language):
    while True:
        print("----")
        stemmer = EnglishStemmer()
        test_input = input()

        #Preprocessing
        tokenized_input = nltk.word_tokenize(test_input, language=language)
        stemmed_input = [stemmer.stem(word.lower()) for word in tokenized_input]

        #---
        input_data = [0] * len(words)
        for word in stemmed_input:
            if word in words:
                input_data[words.index(word)] = 1
        input_data = np.array(input_data).reshape(1, -1)
        predicted_output = model.predict(input_data)[0]
        intent_index = np.argmax(predicted_output)
        response = get_response(intent_index).format(user="dev")
        print(response)

# Training and testing a neural network model chatbot: only sends intents as response.
def train_test():
    #---- Training
    data_de, data_en, data_fr = extract_data()

    #change data and language below
    data = data_de
    lang = "german" #options: french, english, german

    extract_intents(data=data, language=lang)  
    training, output = vectorization(lang)
    model = build_model(training, output)

    #---- Testing
    test_chat(model, lang)
    extract_intents(data=data, language=lang) 

train_test()

