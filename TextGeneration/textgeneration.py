import transformers
from transformers import pipeline

model_name = "bigscience/bloom-1b7"
generator = pipeline("text-generation", model=model_name)

text = "Em sentido estrito, ciência refere-se ao sistema de adquirir conhecimento baseado no método científico."
result = generator(text, max_length=150, do_sample=True)
print(result[0]['generated_text'])