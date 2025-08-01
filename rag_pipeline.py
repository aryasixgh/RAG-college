import ollama
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
# Set the path where ChromaDB is stored. This must match the ingestion script.
PERSIST_DIR = "./chroma_db"
# Choose the Sentence-Transformer model for embeddings. This must match the ingestion script.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Set the Ollama model to use for generating the final answer.
LLM_MODEL = "gemma3:1b"
# Number of relevant chunks to retrieve from the vector store.
K_CHUNKS = 10

def get_rag_answer(question: str) -> str:
    """
    Performs a RAG query by:
    1. Embedding the user question.
    2. Fetching relevant documents from ChromaDB.
    3. Constructing a prompt with the retrieved context.
    4. Calling the Ollama model to generate a final answer.

    Args:
        question (str): The user's question.

    Returns:
        str: The final answer from the LLM.
    """
    # Step 1: Initialize embedding model and ChromaDB
    print("Initializing embedding model and ChromaDB...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": K_CHUNKS})
    except Exception as e:
        return f"Error initializing ChromaDB: {e}. Please run data_ingestion.py first."

    # Step 2: Retrieve relevant documents (chunks)
    print(f"Retrieving top {K_CHUNKS} relevant documents for the question...")
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        print("No relevant documents found in the database. Providing a general answer.")
        context = ""
    else:
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        print("Successfully retrieved documents.")

    # Step 3: Construct the prompt for the LLM
    print("Constructing prompt for the LLM...")
    prompt_template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a helpful assistant. Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer concisely and accurately based on the provided context.

    Context:
    {context}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = prompt_template.format(context=context, question=question)

    # Step 4: Pass the prompt to the Ollama model and get the response
    print("Calling Ollama with the gemma3 model...")
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        final_answer = response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {e}. Is Ollama running and is the '{LLM_MODEL}' model downloaded?"

    return final_answer

if __name__ == "__main__":
    # Check if the ChromaDB directory exists
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: The ChromaDB directory '{PERSIST_DIR}' was not found.")
        print("Please run the 'data_ingestion.py' script first to create the vector store.")
    else:
        print("RAG Pipeline is ready. Enter your question below.")
        while True:
            user_question = input("Your question: ")
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            answer = get_rag_answer(user_question)
            print("\n" + "="*50)
            print("Answer:")
            print(answer)
            print("="*50 + "\n")
