from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
DATA_PATH = "data/ACR_Neurologic"

os.environ['OPENAI_API_KEY'] = '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'

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


# def load_documents():
#     loader = PyPDFLoader(DATA_PATH, glob="*.pdf")
#     documents = loader.load()
#     return documents

def load_documents():
    pdf_files = [file for file in os.listdir(DATA_PATH) if file.endswith('.pdf')]
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        documents.extend(loader.load())
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
