import transformers
from transformers import pipeline

summary = pipeline('summarization')

text = """Jesus, também chamado Jesus de Nazaré (n. 7–2 a.C. – m. 30–33 d.C.) foi um pregador e líder religioso judeu do primeiro século. É a figura central do cristianismo e aquele que os ensinamentos de maior parte das denominações cristãs, além dos judeus messiânicos, consideram ser o Filho de Deus.
O cristianismo e o judaísmo messiânico consideram Jesus como o Messias aguardado no Antigo Testamento e referem-se a ele como Jesus Cristo, um nome também usado fora do contexto cristão.
Praticamente todos os académicos contemporâneos concordam que Jesus existiu realmente, embora não haja consenso sobre a confiabilidade histórica dos evangelhos e de quão perto o Jesus bíblico está do Jesus histórico.
A maior parte dos académicos concorda que Jesus foi um pregador judeu da Galileia, foi batizado por João Batista e crucificado por ordem do governador romano Pôncio Pilatos.
Os académicos construíram vários perfis do Jesus histórico, que geralmente o retratam em um ou mais dos seguintes papéis: o líder de um movimento apocalíptico, o Messias, um curandeiro carismático, um sábio e filósofo, ou um reformista igualitário.
A investigação tem vindo a comparar os testemunhos do Novo Testamento com os registos históricos fora do contexto cristão de modo a determinar a cronologia da vida de Jesus."""

resumo = summary(text, max_length=250, min_length=50)

print(resumo)