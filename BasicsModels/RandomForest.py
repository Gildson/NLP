import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# If want use Google Colab
from google.colab import files
uploaded = files.upload()

spam = pd.read_csv('spam_or_not_spam.csv')

spam.head()

spam.isna().sum()

spam_without_missing = spam.dropna()

spam['label'].value_counts()

previsao = spam_without_missing['email']
classe = spam_without_missing['label']

"""Vetorização"""

vetorizador = TfidfVectorizer()
previsoes = vetorizador.fit_transform(previsao)

print(previsoes.shape)

print(vetorizador.get_feature_names_out()[10:100])

X_train, X_test, y_train, y_test = train_test_split(previsoes, classe, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape)

forest = RandomForestClassifier(n_estimators=500, random_state=42)
forest.fit(X_train, y_train)

_previsoes = forest.predict(X_test)
print(_previsoes)

cm = confusion_matrix(_previsoes, y_test)
plt.figure(figsize = (12, 10))
ax = sns.heatmap(cm, linecolor='white', cmap='crest', linewidth=1, annot=True, fmt='d')
bottom, top = ax.get_ylim()
plt.title('Matriz de Confusão', size=20)
plt.xlabel('Classes Previstas', size=14)
plt.ylabel('Classes Reais', size=14)
plt.show()

accuracy_score(_previsoes, y_test)

print(metrics.classification_report(_previsoes, y_test))