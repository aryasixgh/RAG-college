import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
# Importing the utility to clean up metadata
from langchain_community.vectorstores.utils import filter_complex_metadata
import logging

# --- Setup and Initialization ---

# Configure logging for better visibility into the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (if any)
load_dotenv()

# --- Configuration ---
# Directory containing your PDF files
PDF_DIR = "bajajFinserv2"
# Directory to persist the ChromaDB database
PERSIST_DIR = "chroma_db"
# Name of the collection to store the embeddings
COLLECTION_NAME = "policy_documents_v3"
# Embedding model name
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Main Ingestion Process ---
if __name__ == "__main__":
    logging.info("Starting data ingestion process.")

    # Step 1: Initialize ChromaDB client
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        logging.info("ChromaDB client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB client: {e}")
        exit()

    # Step 2: Check for existing collection and delete if it exists
    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            logging.warning(f"Collection '{COLLECTION_NAME}' already exists. Deleting and recreating.")
            chroma_client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        logging.error(f"Error checking/deleting old collection: {e}")

    # Step 3: Load documents using UnstructuredPDFLoader
    documents = []
    if os.path.exists(PDF_DIR) and os.listdir(PDF_DIR):
        for filename in os.listdir(PDF_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(PDF_DIR, filename)
                logging.info(f"Loading document: {file_path}")
                try:
                    loader = UnstructuredPDFLoader(file_path, mode="elements")
                    documents.extend(loader.load())
                except Exception as e:
                    logging.error(f"Failed to load PDF '{filename}': {e}")
    else:
        logging.error(f"The directory '{PDF_DIR}' does not exist or is empty. Please check the path and contents.")
        exit()

    logging.info(f"Loaded a total of {len(documents)} document elements.")
    
    # Step 3.5: Filter complex metadata to resolve the error
    if documents:
        logging.info("Filtering complex metadata from documents...")
        documents = filter_complex_metadata(documents)
        logging.info("Metadata filtering complete.")

    # Step 4: Initialize the embedding model
    logging.info("Initializing FastEmbed embeddings model...")
    try:
        embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to initialize FastEmbed embeddings: {e}")
        exit()
    logging.info("Embeddings model initialized.")

    # Step 5: Create the vector store and persist it
    if documents:
        logging.info(f"Creating and persisting new collection '{COLLECTION_NAME}'...")
        try:
            db = Chroma.from_documents(
                documents,
                embeddings,
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR
            )
            logging.info(f"Successfully created and persisted collection '{COLLECTION_NAME}'.")
            logging.info(f"Collection count: {db._collection.count()}")
        except Exception as e:
            logging.error(f"Failed to create and persist vector store: {e}")
    else:
        logging.warning("No documents were loaded, skipping vector store creation.")
    
    logging.info("Data ingestion process complete.")
