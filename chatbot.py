import numpy as np
import random
import json
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

import tensorflow as tf

import nltk
nltk.download('punkt')

lenmatizer = WordNetLemmatizer()

intents = json.loads(open('new.json').read())

words =[]
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'s", "'m", "'re", "'ll", "'ve", "'d", "'t"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # add documents in the corpus
        documents.append((word_list, intent['tag']))
        # add to our classes if it's a new tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lenmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(classes))


classes = sorted(set(classes))

pickle.dump(words, open('./words.pkl', 'wb'))
pickle.dump(classes, open('./classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lenmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

print("Training shape:", training.shape)
print("Words shape:", len(words))

# train_x = training[:, len(words)]
# train_y = training[:, len(words):]
training = np.array(training, dtype=object)
train_x = np.array([np.array(i[0], dtype=np.float32) for i in training])
train_y = np.array([np.array(i[1], dtype=np.float32) for i in training])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print("Sample train_x:", train_x[:2])
# print("Sample train_y:", train_y[:2])
# print("train_x dtype:", train_x.dtype)
# print("train_y dtype:", train_y.dtype)
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('./chatbot_model.keras', hist)
print("Model created")



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('new.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('./chatbot_model.keras')

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatize, lower each word and remove duplicates
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # return an array: 0 or 1 for each word in the bag that exists in the sentence
    bag = [0] * len(words)
    sentence_words = clean_up_sentence(sentence)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only differences in the axis


def predict_class(sentence):
    # filter out predictions below a threshold
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    res = softmax(res)
    ERROR_THRESHOLD = 0.5
    results = [
        {"intent": classes[i], "probability": str(prob)} 
               for i, prob in enumerate(res) if prob > ERROR_THRESHOLD
               ]

    # sort by strength of probability
    results.sort(key=lambda x: x["probability"], reverse=True)
    print("Predicted intents:", results)
    return results
    # return_list = []
    # for r in results:
    #     return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return return_list
    

def get_response(intent_list, intents_json):
    if not intent_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    if len(intent_list) > 1 and float(intent_list[0]['probability']) - float(intent_list[1]['probability']) < 0.1:
        return "I'm not sure what you mean. Can you clarify?"
    tag = intent_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response
    # tag = intent_list[0]['intent']
    # list_of_intents = intents_json['intents']
    # for i in list_of_intents:
    #     if i['tag'] == tag:
    #         response = random.choice(i['responses'])
    #         break
    # return response

print("Chatbot is ready to talk!")

while True:
    message = input("You: ")
    if message.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    intent_list = predict_class(message)
    response = get_response(intent_list, intents)
    print("Chatbot:", response)


