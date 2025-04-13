import os
import argparse
from store import UniversityKnowledgeStore

CHROMADB_PATH = "./university_chromadb"
DEFAULT_MANUALS_DOCS_PATH = "./docs"
COLLECTION_NAMES = ["Administrativo", "Estudiante", "Profesor", "Externo"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ingest_from_directory(manuals_docpath):
    # Ingest the university manuals
    files_to_process = {collection: [] for collection in COLLECTION_NAMES}  # Create a dictionary for files to process

    for root, dirs, files in os.walk(manuals_docpath):
        # Exclude the "ignore" subdirectory
        dirs[:] = [d for d in dirs if d != "ignore"]
        
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith((".pdf", ".docx",".txt")):
                # Determine which collection the file belongs to
                for collection in COLLECTION_NAMES:
                    if collection in root:
                        files_to_process[collection].append(file_path)
                    
    for collection, files in files_to_process.items():
        print(f"\n{len(files)} files to process for {collection}")
        if files:
            store.ingest_manuals(files, collection_name=collection)
    
    return store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the ChromaDB database.")
    parser.add_argument("--directory", type=str, default=DEFAULT_MANUALS_DOCS_PATH, help="Directory to read manuals from (default: new_docs)")
    args = parser.parse_args()

    manuals_docpath = args.directory
    print(f"Reading manuals from directory: {manuals_docpath}")

    store = UniversityKnowledgeStore(create=True, path=CHROMADB_PATH)
    ingest_from_directory(manuals_docpath)