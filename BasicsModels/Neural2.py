import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# If want use Google Colab
from google.colab import files
files.upload()

spam = pd.read_csv('spam.csv')
spam.head()

"""Tratamento dos dados"""

labelencoding = LabelEncoder()
y = labelencoding.fit_transform(spam['Category'])

y

mensagens = spam['Message'].values
X_train, X_test, y_train, y_test = train_test_split(mensagens, y, test_size=0.3, random_state=42)

print(X_train)

token = Tokenizer(num_words=1000)
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

print(X_train)

print(f"Tamanho do treinamento: {len(X_train)}")
print(f"Tamanho do teste: {len(X_test)}")

X_train = pad_sequences(X_train, padding="post", maxlen=500)
X_test = pad_sequences(X_test, padding="post", maxlen=500)

print(X_train)

modelo = Sequential()
modelo.add(Embedding(input_dim=len(token.word_index), output_dim=50, input_length=500))
modelo.add(Flatten())
modelo.add(Dense(units=10, activation='relu'))
modelo.add(Dropout(0.1))
modelo.add(Dense(units=1, activation='sigmoid'))

modelo.summary()

modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

modelo.fit(X_train, y_train, epochs=20, batch_size=10, verbose=True, validation_data=(X_test, y_test))

loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

prev = (modelo.predict(X_test) > 0.5)

cm = confusion_matrix(prev, y_test)
plt.figure(figsize = (12, 10))
ax = sns.heatmap(cm, linecolor='white', cmap='crest', linewidth=1, annot=True, fmt='d')
bottom, top = ax.get_ylim()
plt.title('Matriz de Confusão', size=20)
plt.xlabel('Classes Previstas', size=14)
plt.ylabel('Classes Reais', size=14)
plt.show()