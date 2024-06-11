import transformers
from transformers import pipeline

model_name = 'xlm-roberta-large'
unmasker = pipeline('fill-mask', model=model_name)

text = unmasker(["Eu estou sentado na <mask>.","Brasília é a <mask> do Brasil"])
for x in range(len(text)):
  print(text[x])