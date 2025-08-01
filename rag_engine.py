import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# --- Constants ---
PAPERS_DIR = "./papers/"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Knowledge Base Setup ---
def setup_knowledge_base():
    """
    Sets up the ChromaDB knowledge base by loading and vectorizing
    scientific papers from the ./papers/ directory.
    """
    if os.path.exists(DB_DIR):
        print(f"ChromaDB already exists at {DB_DIR}. Skipping setup.")
        return

    print("Knowledge base not found. Setting up...")

    if not os.path.exists(PAPERS_DIR) or not any(f.endswith('.pdf') for f in os.listdir(PAPERS_DIR)):
        print(f"Error: The '{PAPERS_DIR}' directory does not exist or contains no PDF files.")
        print("Please create the directory and add scientific papers in PDF format.")
        # Create the directory if it doesn't exist so the app doesn't crash on first run.
        os.makedirs(PAPERS_DIR, exist_ok=True)
        return

    print("Loading documents...")
    loader = DirectoryLoader(
        PAPERS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    if not documents:
        print("No documents were loaded. Please check the PDF files in the './papers/' directory.")
        return

    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating vector embeddings... (This may take a while)")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Initializing ChromaDB at {DB_DIR}...")
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=DB_DIR
    )
    print("Knowledge base setup complete.")


# --- Query Engine ---
def query_plant_state(user_query: str) -> dict:
    """
    Queries the knowledge base and a local LLM to predict a plant's state.

    Args:
        user_query: The user's full, natural language query.

    Returns:
        A dictionary with "Description" and "Image_Prompt" keys,
        or an error message.
    """
    if not os.path.exists(DB_DIR):
        return {
            "Description": "Error: Knowledge base not found. Please run the setup script first.",
            "Image_Prompt": ""
        }

    print(f"Loading ChromaDB and embedding model for query: '{user_query[:50]}...'")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("Retrieving relevant documents from the knowledge base...")
    retriever = db.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.invoke(user_query)

    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    print("Constructing prompt and querying the local LLM...")

    final_prompt = f"""
You are a botanical scientist and expert in plant physiology, pathology, and genetics.
Your task is to predict the future state of a plant based on a user's query and the provided scientific context.

**Scientific Context from Research Papers:**
---
{context}
---

**User's Query:**
"{user_query}"

**Instructions:**
1.  Analyze the user's query and the scientific context carefully.
2.  Synthesize the information to generate a plausible, science-based forecast of the plant's condition.
3.  Your response MUST be in the following strict format, with no other text or explanations before or after:

Description: [Provide a detailed, scientific description of the plant's predicted state. Mention changes in morphology, health, color, and potential diseases or growth patterns based on the context.]
Image_Prompt: [Create a concise, descriptive prompt for an image generation model to visualize the predicted state. This should be a single line of text. For example: "A close-up of a tomato plant with yellowing leaves and small brown spots, showing signs of blight."]
"""

    try:
        ollama = Ollama(model=LLM_MODEL)
        response_text = ollama.invoke(final_prompt)
        print("LLM response received.")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return {
            "Description": "Error: Could not connect to the local LLM. Is Ollama running?",
            "Image_Prompt": ""
        }

    # --- Parse the LLM Output ---
    description_match = re.search(r"Description:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
    image_prompt_match = re.search(r"Image_Prompt:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)

    description = description_match.group(1).strip() if description_match else "Could not parse description from the model's response."
    image_prompt = image_prompt_match.group(1).strip() if image_prompt_match else "Could not parse image prompt from the model's response."

    return {
        "Description": description,
        "Image_Prompt": image_prompt
    }

if __name__ == '__main__':
    # Example usage for direct testing
    print("Running a test setup and query...")
    # setup_knowledge_base() # Uncomment to run setup
    test_query = "What would happen to a ficus lyrata if it's overwatered for 2 weeks and in a low-light corner?"
    result = query_plant_state(test_query)
    print("\n--- Query Result ---")
    print(f"Description: {result['Description']}")
    print(f"Image Prompt: {result['Image_Prompt']}")
    print("--------------------")
