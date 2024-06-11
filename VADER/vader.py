import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from translate import Translator

sid = SentimentIntensityAnalyzer()

"""Examples that show how vader works"""

sentence = 'I HATE this movie!!!'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = 'I HATE this movie'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = 'I hate this movie'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = 'I hate this movie, but I loved the girl'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = 'This plot is very old'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = ':)'
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = ':('
sentiment = sid.polarity_scores(sentence)
print(sentiment)

sentence = TextBlob('The movie was awesome!')
print(sentence.sentiment)

sentence = TextBlob('I love NY!')
print(sentence.sentiment)

"""How to use the VADER with portuguese language?"""

translator = Translator(from_lang="pt", to_lang='en')

# Translate the text
translate = translator.translate('Isto é um caneta preta')
print(translate)
# Sentiment analysis with VADER
sentiment = sid.polarity_scores(translate)
print(sentiment)