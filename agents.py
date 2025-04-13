from store import UniversityKnowledgeStore

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrievals for ChromaDB
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline


from prompts import PROMPT_ADMINISTRATIVO, PROMPT_ESTUDIANTE, PROMPT_PROFESOR, PROMPT_EXTERNO
import logging
from termcolor import colored


EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
UMBRAL_DE_RELEVANCIA = 0.4
CANTIDAD_DE_CONTEXTOS = 3

# Initialize ChromaDB
chroma_db = "./university_chromadb"
store = UniversityKnowledgeStore(path=chroma_db, create=False)

def get_retriever(llm, collection_name, threshold = UMBRAL_DE_RELEVANCIA, k = CANTIDAD_DE_CONTEXTOS):
    
    logging.info("Initializing retriever")
    
    retriever = store.get_retriever(collection_name=collection_name, threshold=threshold, k = CANTIDAD_DE_CONTEXTOS)
    retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    redundancy_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    logging.info("Retriever initialized successfully")
    
    return compression_retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the subagents
class SpecializedAgent():
    def __init__(self, llm, collection_name, prompt_template):
        super().__init__()
        self.llm = llm
        self.retriever = get_retriever(llm, collection_name=collection_name, threshold=0.2, k=3)
        self.prompt_template = prompt_template
        
    def run(self, query, log_context = False ):
        logging.info("Recuperando Contexto para contestar la pregunta...")
        docs = self.retriever.get_relevant_documents(query)
        context = format_docs(docs)
        if log_context:
            logging.info(colored(f"Contexto recuperado: {context}","blue"))
        prompt = PromptTemplate.from_template(self.prompt_template)
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({"context": context, "query": query})
        return response