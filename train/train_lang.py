import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from data_lang import create_datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

X_train, y_train = create_datasets()
print(len(X_train))

cv = CountVectorizer()
lb = LabelEncoder()

text_train, text_test, label_train, label_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Converts Train and test data to numerical values
X_train = cv.fit_transform(text_train)
X_test = cv.transform(text_test)
y_train = lb.fit_transform(label_train)
y_test = lb.transform(label_test)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Converts numerical values to tensors
X_train_sparse = tf.convert_to_tensor(csr_matrix(X_train).todense(), dtype=tf.float32)
X_val_sparse = tf.convert_to_tensor(csr_matrix(X_val).todense(), dtype=tf.float32)
X_test_sparse = tf.convert_to_tensor(csr_matrix(X_test).todense(), dtype=tf.float32)

print(X_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu', name='L1'),
    tf.keras.layers.Dense(8, activation='relu', name='L2'),
    tf.keras.layers.Dense(len(lb.classes_), activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.summary()

#predictions = model.predict(X_test)
#y_pred = predictions.argmax(axis=1)
#accuracy = accuracy_score(y_test, y_pred)

history = model.fit(X_train_sparse, y_train, epochs=5, batch_size=32, validation_data=(X_val_sparse, y_val))
#model.evaluate(X_test_sparse, y_test, batch_size=16)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#save Model, Countvectorizer and LabelEncoder
model.save("./models/lang_model_v1.keras")
pickle.dump(cv, open("./models/count_vectorizer_v1.pkl", 'wb'))
pickle.dump(lb, open("./models/label_encoder_v1.pkl", 'wb'))


#test function to predict languages based on input
def predict_language_test(input_value):
    input_value = [input_value]
    input_vector = cv.transform(input_value)
    input_sparse = tf.convert_to_tensor(input_vector.todense(), dtype=tf.float32)

    predictions = model.predict(input_sparse)
    predicted_label_index = np.argmax(predictions)
    predicted_language = lb.classes_[predicted_label_index]
    print(predicted_language)

def test():
    while(True):
        input_text = input("---- \n")
        predict_language_test(input_text)

test()
