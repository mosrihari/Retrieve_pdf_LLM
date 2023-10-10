from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def get_llm():
    qa_llm = pipeline(
        task='text2text-generation',
        model=AutoModelForSeq2SeqLM.from_pretrained('LaMini-T5-738M'),
        tokenizer=AutoTokenizer.from_pretrained('LaMini-T5-738M', torch_dtype=torch.float32),
        max_length = 256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    qa_pipeline = HuggingFacePipeline(pipeline=qa_llm)
    return qa_pipeline

def qa_llm():
    my_llm = get_llm()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db =  Chroma(embedding_function=embeddings, persist_directory='db')
    retriever = db.as_retriever()
    qa_retriever = RetrievalQA.from_chain_type(return_source_documents=True, retriever= retriever,
                                chain_type='stuff', llm=my_llm)
    return qa_retriever

def process_instruction(instructions):
    qa = qa_llm()
    results = qa(instructions)
    print(results)
    return 1

if __name__ == '__main__':

    process_instruction('What is fossil fuel emission?')
