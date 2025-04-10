import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

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
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('./chatbot_model.h5', hist)
print("Model created")


