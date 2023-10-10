from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
import chromadb

client = chromadb.PersistentClient('db')
def main():
    for files in os.listdir('docs'):
        pdf_loader = PDFMinerLoader(rf'docs/{files}')
    documents = pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000)
    split_documents = splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents = split_documents, embedding = embeddings,
                          persist_directory='db')
    db.persist()
    db = None

if __name__ == '__main__':
    main()