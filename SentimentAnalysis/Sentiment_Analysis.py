import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

# If use the code in the Google Colab this two lines is important to import the dataset
from google.colab import files
files.upload()

Tweets = pd.read_csv('Tweets.csv')
Tweets.head()

Tweets.groupby('airline_sentiment').size()

Tweets = Tweets[Tweets.airline_sentiment_confidence > 0.8]

token = Tokenizer(num_words=100)
token.fit_on_texts(Tweets.text.values)
X = token.texts_to_sequences(Tweets.text.values)
X = pad_sequences(X, padding="post", maxlen=100)
print(X)

labelencoding = LabelEncoder()
Y = labelencoding.fit_transform(Tweets.airline_sentiment)
Y = np_utils.to_categorical(Y)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape )

modelo = Sequential()
modelo.add(Embedding(input_dim=len(token.word_index), output_dim=128, input_length=X.shape[1]))
modelo.add(SpatialDropout1D(0.2))
modelo.add(LSTM(units=196, dropout=0.2, recurrent_dropout=0, activation='tanh', recurrent_activation='sigmoid',unroll=False, use_bias=True))
modelo.add(Dense(3, activation='softmax'))

modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(modelo.summary())

modelo.fit(X_train, y_train, epochs=10, batch_size=30, verbose=True, validation_data=(X_test, y_test))

loss, accuracy = modelo.evaluate(X_test, y_test, verbose=False)
print("Test loss: {:.4f}".format(loss))
print("Test Accuracy: {:.4f}".format(accuracy))

predictions = modelo.predict(X_test)
predictions = predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
predictions = (labelencoding.inverse_transform((predictions)))
predictions = pd.DataFrame({'Classes Previstas': predictions})
predictions.head()

actual = y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (labelencoding.inverse_transform((actual)))
actual = pd.DataFrame({'Classes Reais': actual})
actual.head()