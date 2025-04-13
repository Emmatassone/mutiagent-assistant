# Import ChomaDB to store the data
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
# Import langchain functions to embed the data
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Other tools
from tqdm import tqdm
from termcolor import colored

class UniversityKnowledgeStore:
    CHROMADB_PATH = "./chromadb" 
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAMES = ["Administrativo", "Estudiante", "Profesor", "Externo"]
    
    def __init__(self, create=False, path=None) -> None:
        self.chromadb_path = path if path else self.CHROMADB_PATH
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDINGS_MODEL)
        
        # Initialize the Chroma client
        self.client = chromadb.PersistentClient(
            path=self.chromadb_path,
            settings=Settings(allow_reset=True)
        )

        # Use the updated HuggingFaceEmbeddings from langchain_huggingface
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDINGS_MODEL)

        print(colored(f"ChromaDB path: {self.chromadb_path}", "blue"))
        print(colored(f"Embeddings model: {self.EMBEDDINGS_MODEL}", "blue"))
        print(colored(f"Create vector dataset is set to {create}.", "green"))
        
        if create:
            print(colored("Deleting existing collections ...", "red"))
            self.client.reset()
        
        
        self.vector_dbs = []
        # Create a collection for each type of user
        for collection_name in self.COLLECTION_NAMES:
            self.client.get_or_create_collection(
                collection_name,
                embedding_function=self.sentence_transformer_ef
            )
            
            self.vector_dbs.append(
                Chroma(
                    client=self.client,
                    collection_name=collection_name,
                    persist_directory=self.chromadb_path,
                    embedding_function=self.embeddings
                )
            )
            
        print(colored(f"ChromaDB initialized with {len(self.COLLECTION_NAMES)} collections.", "blue"))
        for collection_name in enumerate(self.COLLECTION_NAMES):
            print(colored(f"Collection '{collection_name}' has {self.vector_dbs[0]._collection.count()} chunks.", "green"))          
        
    @staticmethod
    def _process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in tqdm(pdf_files, "PDFs"):
            try:
                pdf_loader = PyPDFLoader(pdf_file_path)
                batch_docs.extend(pdf_loader.load())
            except Exception as e:
                print(colored(f"Error processing PDF file '{pdf_file_path}': {e}","red"))

        text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name=UniversityKnowledgeStore.EMBEDDINGS_MODEL))
        pdf_chunks = text_splitter.split_documents(batch_docs)

        return pdf_chunks
    
    @staticmethod
    def _process_docx_batch(docx_files):
        batch_docs = []
        for docx_file_path in tqdm(docx_files, "DOCX files"):
            try:
                docx_loader = UnstructuredWordDocumentLoader(docx_file_path)
                batch_docs.extend(docx_loader.load())
            except Exception as e:
                print(colored(f"Error processing DOCX file '{docx_file_path}': {e}","red"))

        text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name=UniversityKnowledgeStore.EMBEDDINGS_MODEL))
        docx_chunks = text_splitter.split_documents(batch_docs)

        return docx_chunks
    
    @staticmethod
    def _process_txt_batch(txt_files):
        batch_docs = []
        for txt_file_path in tqdm(txt_files, "TXT files"):
            try:
                txt_loader = TextLoader(txt_file_path)
                batch_docs.extend(txt_loader.load())
            except Exception as e:
                print(colored(f"Error processing DOCX file '{txt_file_path}': {e}","red"))

        text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name=UniversityKnowledgeStore.EMBEDDINGS_MODEL))
        txt_chunks = text_splitter.split_documents(batch_docs)

        return txt_chunks

    def ingest_manuals(self, files: list[str], collection_name: str):
        
        if collection_name not in self.COLLECTION_NAMES:
            print(colored(f"Error: Collection {collection_name} Not Found", "red"))
            return
        
        print(colored("Ingesting documents from document database ...", "blue"))
        
        pdf_files = [file for file in files if file.lower().endswith(".pdf")]
        docx_files = [file for file in files if file.lower().endswith(".docx")]
        txt_files = [file for file in files if file.lower().endswith(".txt")]
        
        print(colored(f"Number of PDF files: {len(pdf_files)}", "blue"))
        print(colored(f"Number of DOCX files: {len(docx_files)}", "blue"))
        print(colored(f"Number of TXT files: {len(txt_files)}", "blue"))
        
        pdf_chunks = []
        docx_chunks = []
        txt_chunks = []
        
        if len(pdf_files) > 0:
            pdf_chunks = self._process_pdf_batch(pdf_files)
            print(colored(f"Number of chunks from PDF files: {len(pdf_chunks)}", "green"))
        
        if len(docx_files) > 0:
            docx_chunks = self._process_docx_batch(docx_files)
            print(colored(f"Number of chunks from DOCX files: {len(docx_chunks)}", "green"))
        
        if len(txt_files) > 0:
            txt_chunks = self._process_txt_batch(txt_files)
            print(colored(f"Number of chunks from TXT files: {len(txt_chunks)}", "green"))
        
        all_chunks = pdf_chunks + docx_chunks + txt_chunks
        print(colored(f"Total number of chunks to be ingested: {len(all_chunks)}", "blue"))
        
        if len(all_chunks) > 0:
            for i, collection in enumerate(self.COLLECTION_NAMES):
                if collection_name == collection:
                    self.vector_dbs[i].add_documents(all_chunks)
                    print(colored(f"Ingestion of {collection_name} Completed.", "green"))
                    return
        else:
            print(colored("No chunks to ingest.", "yellow"))
    
    def search(self, query: str, collection_name: str, k: int = 3):
        
        if collection_name not in self.COLLECTION_NAMES:
            print(colored(f"Error: Collection {collection_name} Not Found", "red"))
            return []
        
        for i, collection in enumerate(self.COLLECTION_NAMES):
            if collection_name == collection:
                docs = self.vector_dbs[i].similarity_search(query, k=k)
                return docs
    
    def _get_database(self, collection_name: str):
        for i, collection in enumerate(self.COLLECTION_NAMES):
            if collection_name == collection:
                return self.vector_dbs[i]
        
    
    def get_retriever(self, collection_name: str = None, threshold: float = 0.5, k: int = 3):
        retriever = None
        if collection_name not in self.COLLECTION_NAMES:
            print(colored(f"Error: collection {collection_name} Not Found","red"))
            return None
        
        chroma_db = self._get_database(collection_name=collection_name)
        retriever = chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold, "k": k}
        ) 
    
        return retriever