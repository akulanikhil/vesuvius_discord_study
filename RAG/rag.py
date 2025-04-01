from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from uuid import uuid4

embedding_model = HuggingFaceEmbeddings()
database_loc = ("./chroma_db_test1")

vectorstore = Chroma(persist_directory=database_loc,
      embedding_function=embedding_model)

from typing import List
from langchain_core.runnables import chain
from langchain_core.documents import Document

@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query, k=8))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs

phrase = "How are the scrolls scanned?"
embedding = HuggingFaceEmbeddings().embed_query(phrase)

results = retriever.invoke(phrase)
print(results)

