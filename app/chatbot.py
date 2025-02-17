import re
import nltk
import random
import pickle
import json
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, GermanStemmer, FrenchStemmer
from intents import intent_classifier

nltk.download('punkt_tab')
stopwords = []
intents = []
words= []
classes= []
stemmer = []

#Load the language detection & Chat models
model_lang = tf.keras.models.load_model("./models/language/lang_model_v1.keras")
model_en = tf.keras.models.load_model("./models/chatbot/m_en_chatbot_v1.keras") 
model_de = tf.keras.models.load_model("./models/chatbot/m_de_chatbot_v1.keras")
model_fr = tf.keras.models.load_model("./models/chatbot/m_fr_chatbot_v1.keras")

#Load the Dialogue Data
with open("./datasets/reference_en_v1.json", "r") as file:
    data_en = json.load(file)

with open("./datasets/reference_de_v1.json", "r", encoding='utf_8_sig') as file:
    data_de = json.load(file)

with open("./datasets/reference_fr_v1.json", 'r', encoding='utf_8_sig') as file:
    data_fr = json.load(file)

intents_en = data_en["intents"]
intents_de = data_de["intents"]
intents_fr = data_fr["intents"]

#Load the Chatbot models
data_en = pickle.load(open("./models/chatbot/en_dialogue_v1.pkl", "rb"))
data_de = pickle.load(open("./models/chatbot/de_dialogue_v1.pkl", "rb"))
data_fr = pickle.load(open("./models/chatbot/fr_dialogue_v1.pkl", "rb"))

words_en = data_en["words"]
words_de = data_de["words"]
words_fr = data_fr["words"]
classes_en = data_en["classes"]
classes_de = data_de["classes"]
classes_fr = data_fr["classes"]

#Initialize Encoder and Labeler
cv = pickle.load(open("./models/language/count_vectorizer_v1.pkl", "rb"))
lb = pickle.load(open("./models/language/label_encoder_v1.pkl", "rb"))

def predict_language(input_text):
    # Preprocess input text
    input_text = [input_text]
    input_vector = cv.transform(input_text)
    input_sparse = tf.convert_to_tensor(input_vector.todense(), dtype=tf.float32)

    # Make prediction using the trained model
    predictions = model_lang.predict(input_sparse)

    # Convert predicted index back to language label
    predicted_label_index = np.argmax(predictions)
    predicted_language = lb.classes_[predicted_label_index]
    return predicted_language

# acquire random response from the recognized intent
def get_response(intent_index):
    global intents
    recognized_intent = classes[intent_index]
    for intent in intents:
        if intent["tag"] == recognized_intent:
            response = random.choice(intent["responses"])
            break
            

    return response #callable function, analyze and send input back

def set_language(user_input):
    global intents
    global words
    global classes
    global stemmer
    global stopwords
    set_language = 0 #sets language setting for tokenization and Stemming

    lang = predict_language(user_input)
    print(lang)
    match lang:
        case 'eng':
            model = model_en
            model.name = "Eras"
            stemmer = EnglishStemmer()
            stopwords = nltk.corpus.stopwords.words('english')
            intents = intents_en
            words = words_en
            classes = classes_en
            set_language = "english"
        case 'deu':
            model = model_de
            model.name = "Deras"
            stemmer = GermanStemmer()
            stopwords = nltk.corpus.stopwords.words('german')
            intents = intents_de
            words = words_de
            classes = classes_de
            set_language = "german"
        case 'fra':
            model = model_fr
            model.name = "Feras"
            stemmer = FrenchStemmer()
            stopwords = nltk.corpus.stopwords.words('french')
            intents = intents_fr
            words = words_fr
            classes = classes_fr
            set_language = "french"
        case _:
            model = None
    return model, set_language

#Breaksdown the user input into chunks, removing stopwords
def preprocess_user_input(user_input, set_language):
    triggerList = ["Nova", "nova"] # Trigger names, the AI will only respond when "Nova" is also in the input either voice or Text

    user_input = re.sub('[!?,.-]','', user_input)
    tokenized_input = nltk.word_tokenize(user_input, language=set_language.lower())

    if any(trigger in tokenized_input for trigger in triggerList):
        print(tokenized_input)
        for word in tokenized_input:
            if word.lower() in stopwords:
                tokenized_input.remove(word)
            print(f"Stopwords filtered: {tokenized_input}")
            stemmed_input = [stemmer.stem(word.lower()) for word in tokenized_input]
            print(f"Filtered input: {stemmed_input}")
            return stemmed_input
    else:
        return False
    
    
#Analyzes user input, outputs an context correct response
def analyze_user_input(username, user_input, set_model, set_language):
    preprocessed_input = preprocess_user_input(user_input, set_language)
    
    if preprocessed_input == False:
        return "..."
            
    input_data = [0] * len(words)
    for word in preprocessed_input:
        if word in words:
            input_data[words.index(word)] = 1
    input_data = np.array(input_data).reshape(1, -1)
    predicted_output = set_model.predict(input_data)[0]
    intent_index = np.argmax(predicted_output)
    print(classes[intent_index])
    email_data = intent_classifier(user_input, classes[intent_index], set_language.lower()) #Checks if intent is connected with a task
    if email_data == False:
        return
    else:
        response = get_response(intent_index).format(user=username, email='classified', sender=email_data[0],
                                                        subject=email_data[1], snippet=email_data[2], count=email_data[0])
        return response

#intent_classifier(user_input, classes[intent_index]) # Checks if intent is connected with a task