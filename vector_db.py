import chromadb
from chromadb.utils import embedding_functions
import os
import json

# Define where the vector database will store its data
DB_PATH = "database/vector_store"

class VectorDB:
    """A wrapper for ChromaDB to handle semantic search for document pages."""
    
    def __init__(self, collection_name="document_metadata"):
        # Ensure the storage directory exists
        os.makedirs(DB_PATH, exist_ok=True)
        
        # Initialize the persistent client
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # Load a default embedding function 
        # (This uses 'all-MiniLM-L6-v2' by default, which is efficient for local use)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or retrieve the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.ef
        )

    def index_document(self, doc_id, text_to_embed, metadata):
        """
        Index a document page's metadata for semantic search.
        
        Args:
            doc_id (str): Unique ID for the document page (e.g., 'doc_page_1').
            text_to_embed (str): The descriptive text we want to search over.
            metadata (dict): The structured metadata from SQLite.
        """
        # ChromaDB expects metadata values to be simple types (str, int, float, bool)
        # We sanitize the metadata dict just in case
        sanitized_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized_metadata[k] = v
            else:
                sanitized_metadata[k] = str(v)

        self.collection.add(
            ids=[str(doc_id)],
            documents=[text_to_embed],
            metadatas=[sanitized_metadata]
        )

    def semantic_search(self, query, n_results=5):
        """
        Search for documents by semantic meaning.
        
        Returns:
            list: A list of result objects containing metadata and scores.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Reformat results into a cleaner list of dicts
        search_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                search_results.append({
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "document": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        return search_results

if __name__ == "__main__":
    # Quick test
    vdb = VectorDB()
    test_id = "test_invoice_001"
    test_text = "Medical invoice from General Hospital for surgery on 2024-05-12, amount $1200"
    test_meta = {"vendor": "General Hospital", "amount": 1200.0, "category": "medical"}
    
    vdb.index_document(test_id, test_text, test_meta)
    
    print("Searching for 'healthcare costs'...")
    results = vdb.semantic_search("healthcare costs")
    for r in results:
        print(f"Match: {r['document']} (Dist: {r['distance']})")
