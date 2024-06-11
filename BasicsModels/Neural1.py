import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

vetorizador = CountVectorizer()
vetorizador.fit(X_train)
X_Train = vetorizador.transform(X_train)
X_Test = vetorizador.transform(X_test)

print(X_Train.shape, X_Test.shape)

modelo = Sequential()
modelo.add(Dense(units=10, activation='relu', input_dim=X_Train.shape[1]))
modelo.add(Dropout(0.1))
modelo.add(Dense(units=8, activation='relu'))
modelo.add(Dropout(0.1))
modelo.add(Dense(units=1, activation='sigmoid'))

modelo.summary()

modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

modelo.fit(X_Train, y_train, epochs=20, batch_size=10, verbose=True, validation_data=(X_Test, y_test))

loss, accuracy = modelo.evaluate(X_Test, y_test)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

prev = (modelo.predict(X_Test) > 0.5)

cm = confusion_matrix(prev, y_test)
plt.figure(figsize = (12, 10))
ax = sns.heatmap(cm, linecolor='white', cmap='crest', linewidth=1, annot=True, fmt='d')
bottom, top = ax.get_ylim()
plt.title('Matriz de Confusão', size=20)
plt.xlabel('Classes Previstas', size=14)
plt.ylabel('Classes Reais', size=14)
plt.show()