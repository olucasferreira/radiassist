import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import chromadb

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = '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'

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join('', 'db.sqlite3'),
#     }
# }

# CHROMA_PATH = "chroma"

PROMPT_TEMPLATE='Answer the question based on the above context: {question}   \n\n\n {context}'

# PROMPT_TEMPLATE="""

# Contexto: Esta rede de linguagem foi treinada para avaliar solicitações de exames de imagem médica com base nas diretrizes da American College of Radiology (ACR). A rede foi treinada para fornecer recomendações sobre a apropriação do exame solicitado e, se necessário, sugerir um exame mais adequado, explicando o motivo e referenciando as fontes (arquivos PDF) das diretrizes da ACR.

# Com base nas diretrizes da American College of Radiology (ACR) e na análise das informações fornecidas, esta rede de linguagem oferece as seguintes recomendações, no formato a seguir em português:

# 1. **Análise de Pedido de Exame:**
#    - **Hipótese Diagnóstica:** [Inserir hipótese diagnóstica com base no texto de entrada]
#    - **Exame Indicado:** [Inserir tipo de exame recomendado]
 
# 2. **Explicação Detalhada:**
#    - [Explicação detalhada sobre a apropriação do exame selecionado ou sobre por que o exame sugerido é mais apropriado]. Esta recomendação é fundamentada nas diretrizes da ACR, conforme documentado no arquivo PDF referenciado [Inserir nome do arquivo PDF e número da seção, se aplicável].

# ----------------------------------
# \n\n\n {context}
# """


def main(query_text):
    # CHROMA_PATH = "chroma"
    parser = argparse.ArgumentParser()
    query_text = query_text


    # client = chromadb.PersistentClient(path="chroma")

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    # db = Chroma(client=client, embedding_function=embedding_function)
    db = Chroma(persist_directory='chroma', embedding_function=embedding_function)
    db.persist()
    print(db)
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(results)
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)
    print(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"{response_text.content}\n\n\n"

    return formatted_response, sources


if __name__ == "__main__":
    main()
