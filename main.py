import csv
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical

fileName = 'tweets.csv'
tweets, labels = [], []
tweets_text = []
stop_words = stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text_clean = ""
    for w in text.split():
        if w not in stop_words:
            text_clean += w + " "
    print("text", text)
    return ''.join([c for c in text_clean if c not in punctuation])

def convert(x):
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    word_frequency = ""
    
    for row in csv_reader:
        # if line_count == 0:
        #     print(f' Column names are {", ".join(row)}')
        #     line_count += 1
        # else:
        # print(row)
        
        if line_count !=0:
            processed_text = preprocess_text(row[10])
            tweet = {
                "id" : row[0],
                "raw_text" : row[10],
                "text" : processed_text,
                "label" : row[1]
            }
            labels.append(row[1])
            word_frequency += " " + processed_text
            tweets.append(tweet)
            tweets_text.append(processed_text)


        # if line_count == 1000:
        #     break
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
        line_count += 1

word_frequency = word_frequency.split()
words_all = word_frequency
# word_frequency = Counter(word_frequency)

encoded_labels = [1 if label =='positive' else 0 for label in labels]
encoded_labels = np.array(encoded_labels)

tokenizer = Tokenizer(num_words = 10000, split=' ')
tokenizer.fit_on_texts(tweets_text)
# print(words_all)
# print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(tweets_text)
print(X[0])
X = pad_sequences(X)
print(X.shape)

print(X[0])

Y = pd.get_dummies(labels).values

embed_dim = 128
lstm_out = 200
batch_size = 32



model = Sequential()
model.add(Embedding(10000, embed_dim,input_length = X.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(3, activation="softmax"))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
print(labels)
print(Y)

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)
print("TTT", X_train, Y_train)
model.fit(X_train, Y_train, batch_size =batch_size, epochs = 10,  verbose = 5)


score,acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))
print(line_count)