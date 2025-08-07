import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_DIR = "./bajajFinserv"
PERSIST_DIR = "./chroma_db" # Chroma DB Storage location
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def ingest_data():
    """
    Ingests data from PDF files, splits them into chunks, creates embeddings,
    and stores them in a ChromaDB vector store.
    """
    print("Starting data ingestion process...")

    # Step 1: Load documents from the PDF directory
    print(f"Loading documents from '{PDF_DIR}'...")
    documents = []
    # Use glob to find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIR}'. Please add some PDFs and try again.")
        return

    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"  - Loaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  - Failed to load {os.path.basename(file_path)}: {e}")
            
    if not documents:
        print("No documents could be loaded. Exiting.")
        return

    # Step 2: Split documents into smaller, manageable chunks
    print(f"Splitting documents into chunks of size {CHUNK_SIZE} with overlap {CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]  # A good default list of separators
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  - Created {len(chunks)} chunks.")

    # Step 3: Create embeddings using HuggingFace's model
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Step 4: Store chunks and embeddings in ChromaDB
    print(f"Creating and persisting ChromaDB vector store at '{PERSIST_DIR}'...")
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR
        )
        db.persist()
        print("  - ChromaDB created successfully.")
        print("Data ingestion complete!")
    except Exception as e:
        print(f"An error occurred while creating or persisting ChromaDB: {e}")

if __name__ == "__main__":
    # Ensure the downloaded_pdfs directory exists
    if not os.path.exists(PDF_DIR):
        print(f"The directory '{PDF_DIR}' does not exist. Creating it now.")
        os.makedirs(PDF_DIR)
        print("Please place your PDF files in this directory and re-run the script.")
    else:
        ingest_data()
