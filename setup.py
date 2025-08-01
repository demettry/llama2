from rag_engine import setup_knowledge_base
import time

if __name__ == '__main__':
    print("Starting the knowledge base setup process...")
    start_time = time.time()

    # This function is imported from the backend engine.
    # It will handle loading PDFs, chunking, embedding, and storing them.
    setup_knowledge_base()

    end_time = time.time()
    print(f"Knowledge base setup finished in {end_time - start_time:.2f} seconds.")
    print("You can now run the main application: streamlit run project_genesis_app.py")
