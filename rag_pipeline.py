import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from langchain_community.embeddings import FastEmbedEmbeddings
import logging

# --- Setup and Initialization ---

# Configure logging for better visibility into the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()
api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")

# Check if the API key is available
if not api_key_deepseek:
    logging.error("DEEPSEEK_API_KEY not found in environment variables. Please check your .env file.")
    exit()

# Set up the API client
try:
    # Switched to a model known to be available on OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key_deepseek,
    )
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    exit()

# Initialize the FastEmbed model for embeddings
logging.info("Loading FastEmbed model...")
try:
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    logging.info("FastEmbed model loaded.")
except Exception as e:
    logging.error(f"Failed to load FastEmbed model: {e}")
    exit()

# --- RAG Pipeline Refinement ---

def generate_sub_queries(user_question: str) -> list[str]:
    """
    Uses the LLM to generate more specific and robust sub-queries based on the original question.
    """
    logging.info("Generating sub-queries...")
    sub_query_prompt = f"""
    You are a query generation expert. Your task is to take a single user question about an insurance policy and break it down into 3 to 5 highly specific and effective search queries. These queries should be designed to retrieve the most relevant information from a technical document.

    User Question: {user_question}

    Provide your queries as a comma-separated list, without any extra text or numbering.
    For example: "grace period for premium payment, due date for premium, policy renewal grace period"
    """

    try:
        completion = client.chat.completions.create(
            # Using a working model with a large context window
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "user", "content": sub_query_prompt}],
            max_tokens=250
        )
        response_text = completion.choices[0].message.content
        sub_queries = [q.strip() for q in response_text.split(',') if q.strip()]
        logging.info(f"Generated sub-queries: {sub_queries}")
        return sub_queries
    except Exception as e:
        logging.error(f"Error generating sub-queries: {e}")
        return [user_question]

def retrieve_and_synthesize_context(queries: list[str], collection) -> str:
    """
    Performs retrieval using all generated queries and synthesizes the results.
    """
    full_context = ""
    unique_contexts = set()

    logging.info(f"Retrieving context for {len(queries)} queries...")
    for query in queries:
        try:
            query_embedding = embedding_model.embed_query(query)
            
            # Keeping n_results at 3 for more context
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            retrieved_docs = results['documents'][0]
            logging.info(f"Retrieved documents for query '{query}': {retrieved_docs}")

            for doc in retrieved_docs:
                unique_contexts.add(doc)
        except Exception as e:
            logging.error(f"Error during retrieval for query '{query}': {e}")
            
    # This section has been updated with a try-except block
    try:
        full_context = "\n\n---\n\n".join(list(unique_contexts))
        logging.info("Context retrieval and synthesis complete.")
    except Exception as e:
        logging.error(f"Error during context synthesis: {e}")
        full_context = "Error synthesizing context."

    return full_context

def answer_question_with_context(question: str, context: str) -> str:
    """
    Uses the final prompt template to get a concise and direct answer from the LLM.
    """
    logging.info("Generating final answer...")
    final_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an expert insurance policy document assistant. Provide a concise and accurate answer to the user's question based ONLY on the provided context.
    - Start with 'Yes' or 'No' if it is a direct yes/no question.
    - Extract all specific numbers, percentages, and conditions directly from the text.
    - If the answer is descriptive, provide a brief, well-structured summary.
    - If the context does not contain the answer, state that the information is not available in the document.

    Context:
    {context}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Question: {question}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    """
    
    try:
        completion = client.chat.completions.create(
            # Using a working model with a large context window
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "user", "content": final_prompt}],
            # Keeping max_tokens at 100 for conciseness
            max_tokens=100
        )
        answer = completion.choices[0].message.content.strip()
        logging.info("Final answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error generating final answer with the LLM: {e}")
        return "An error occurred while trying to answer the question."

def full_rag_pipeline(user_question: str, collection) -> str:
    """
    Orchestrates the entire refined RAG process.
    """
    sub_queries = generate_sub_queries(user_question)
    context = retrieve_and_synthesize_context(sub_queries, collection)
    answer = answer_question_with_context(user_question, context)
    return answer

# --- Main execution block ---
if __name__ == "__main__":
    try:
        chroma_client = chromadb.PersistentClient(path="chroma_db/")
        logging.info("ChromaDB client initialized from local directory.")
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB: {e}")
        exit()
    
    collection_name = "policy_documents_v3"
    
    try:
        available_collections = chroma_client.list_collections()
        found_collections = [c.name for c in available_collections]
        logging.info(f"Found the following collections: {found_collections}")
        
        if collection_name not in found_collections:
            logging.error(f"The selected collection '{collection_name}' was not found. Please check the list of found collections and update 'collection_name' accordingly.")
            exit()
        
        collection = chroma_client.get_collection(collection_name)
    
        if collection.count() == 0:
            logging.error(f"The collection '{collection_name}' is empty. This is likely because the data ingestion script was not run successfully.")
            logging.error("Please run `data_ingestion.py` first to populate the database, then run this script again.")
            exit()
    except Exception as e:
        logging.error(f"An error occurred while accessing the ChromaDB collection: {e}")
        exit()
    
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    print("\n--- Starting RAG Query Process ---")
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1}: {q}")
        response = full_rag_pipeline(q, collection)
        print(f"Answer: {response}")
        print("-----------------------------------")
    print("--- RAG Query Process Complete ---")
