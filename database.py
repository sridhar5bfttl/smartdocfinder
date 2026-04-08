import sqlite3
import json
import os
from datetime import datetime
from vector_db import VectorDB

DB_PATH = "database/doc_finder.db"

def init_db():
    """Initialize the SQLite database and create the documents table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create documents table with fields for extracted metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_pdf TEXT NOT NULL,
            image_path TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            metadata_json TEXT,
            doc_date TEXT,
            amount REAL,
            vendor TEXT,
            purpose TEXT,
            doc_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def insert_document(pdf_path, image_path, page_num, metadata):
    """Insert a processed document page into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    metadata_json = json.dumps(metadata)
    
    cursor.execute('''
        INSERT INTO documents (
            original_pdf, image_path, page_number, metadata_json, 
            doc_date, amount, vendor, purpose, doc_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pdf_path, image_path, page_num, metadata_json,
        metadata.get('date'), 
        metadata.get('amount'), 
        metadata.get('vendor'), 
        metadata.get('purpose'),
        metadata.get('doc_type')
    ))
    
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    
    # --- Vector Indexing ---
    vdb = VectorDB()
    # Create a descriptive text block for the semantic index
    description = (
        f"Document: {pdf_path} (Page {page_num}). "
        f"Vendor: {metadata.get('vendor', 'Unknown')}. "
        f"Amount: ${metadata.get('amount', 0.0)}. "
        f"Date: {metadata.get('date', 'Unknown')}. "
        f"Purpose: {metadata.get('purpose', 'N/A')}. "
        f"Type: {metadata.get('doc_type', 'N/A')}."
    )
    # Use SQLite ID to link back (Ensure it's a string, but without the 'db_' prefix)
    vdb.index_document(str(doc_id), description, metadata)

def get_all_documents():
    """Retrieve all processed documents from the database."""
    conn = sqlite3.connect(DB_PATH)
    # Use Row factory for easier dictionary access
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents ORDER BY created_at DESC')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def reindex_all_for_vectors():
    """Wipe ChromaDB and re-index everything from SQLite with standardized IDs."""
    docs = get_all_documents()
    vdb = VectorDB()
    try:
        # Delete existing collection to avoid ID confusion
        vdb.client.delete_collection("document_metadata")
        # Re-init to recreate collection
        vdb.__init__()
        
        indexed_count = 0
        for doc in docs:
            # Reconstruct the search description
            description = (
                f"Document: {doc['original_pdf']} (Page {doc['page_number']}). "
                f"Vendor: {doc['vendor'] or 'Unknown'}. "
                f"Amount: ${doc['amount'] or 0.0}. "
                f"Date: {doc['doc_date'] or 'Unknown'}. "
                f"Purpose: {doc['purpose'] or 'N/A'}."
            )
            # Use raw string of SQLite ID
            vdb.index_document(str(doc['id']), description, json.loads(doc['metadata_json']))
            indexed_count += 1
        return indexed_count
    except Exception as e:
        print(f"Re-indexing failed: {e}")
        return 0

def get_documents_by_ids(doc_ids):
    """Retrieve specific documents by their SQLite IDs, preserving order."""
    if not doc_ids:
        return []
        
    # Standardize IDs to integers immediately
    clean_ids = []
    for did in doc_ids:
        try:
            if isinstance(did, str) and "db_" in did:
                clean_ids.append(int(did.replace("db_", "")))
            else:
                clean_ids.append(int(did))
        except (ValueError, TypeError):
            continue

    if not clean_ids:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(clean_ids))
    query = f'SELECT * FROM documents WHERE id IN ({placeholders})'
    
    cursor.execute(query, clean_ids)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Map results by ID for quick lookup to restore order
    doc_map = {row['id']: row for row in rows}
    
    # Return documents in the order they appeared in clean_ids (the semantic order)
    return [doc_map[did] for did in clean_ids if did in doc_map]

def update_document_metadata(doc_id, metadata):
    """Update a document's metadata in both SQLite and Vector DB, and re-index."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    metadata_json = json.dumps(metadata)
    
    cursor.execute('''
        UPDATE documents 
        SET metadata_json = ?, doc_date = ?, amount = ?, vendor = ?, purpose = ?, doc_type = ?
        WHERE id = ?
    ''', (
        metadata_json, 
        metadata.get('date'), 
        metadata.get('amount'), 
        metadata.get('vendor'), 
        metadata.get('purpose'), 
        metadata.get('doc_type'),
        doc_id
    ))
    
    # Fetch original details to reconstruct the vector description
    cursor.execute("SELECT original_pdf, page_number FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()
    
    if row:
        pdf_name, page_num = row
        # Update Vector DB (Upsert)
        try:
            vdb = VectorDB()
            description = (
                f"Document: {pdf_name} (Page {page_num}). "
                f"Vendor: {metadata.get('vendor', 'Unknown')}. "
                f"Amount: ${metadata.get('amount', 0.0)}. "
                f"Date: {metadata.get('date', 'Unknown')}. "
                f"Purpose: {metadata.get('purpose', 'N/A')}."
            )
            vdb.index_document(str(doc_id), description, metadata)
        except Exception as e:
            print(f"Vector update failed: {e}")
            
    return True

def delete_document(doc_id):
    """Remove a document from both SQLite and Vector DB, and clean up the file."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Identify image path for disk cleanup
    cursor.execute("SELECT image_path FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    if row:
        img_path = row[0]
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"File deletion failed: {e}")
                
    # Delete from SQLite
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    # Delete from ChromaDB
    try:
        vdb = VectorDB()
        vdb.collection.delete(ids=[str(doc_id)])
    except Exception as e:
        print(f"Vector deletion failed: {e}")

def cleanup_missing_documents():
    """Purge all database records where the associated image file is no longer on disk."""
    docs = get_all_documents()
    deleted_count = 0
    for doc in docs:
        if not os.path.exists(doc['image_path']):
            # Call our delete logic (will skip the file delete part since it doesn't exist)
            delete_document(doc['id'])
            deleted_count += 1
    return deleted_count

def search_documents(query=None, start_date=None, end_date=None, min_amount=None, max_amount=None, categories=None):
    """
    Perform a flexible search across SQLite metadata using multiple filters.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    sql = "SELECT * FROM documents WHERE 1=1"
    params = []
    
    if query:
        sql += " AND (original_pdf LIKE ? OR vendor LIKE ? OR purpose LIKE ?)"
        query_param = f"%{query}%"
        params.extend([query_param, query_param, query_param])
    
    if start_date:
        sql += " AND doc_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND doc_date <= ?"
        params.append(end_date)
    
    if min_amount is not None:
        sql += " AND amount >= ?"
        params.append(min_amount)
    if max_amount is not None:
        sql += " AND amount <= ?"
        params.append(max_amount)
        
    if categories:
        # categories should be a list of strings
        placeholders = ",".join(["?"] * len(categories))
        sql += f" AND purpose IN ({placeholders})"
        params.extend(categories)
        
    sql += " ORDER BY created_at DESC"
    cursor.execute(sql, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def clear_all_documents():
    """Wipe the database and vector store for a fresh start."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    
    # Also clear Vector DB
    from vector_db import VectorDB
    vdb = VectorDB()
    # ChromaDB collections can be deleted or emptied
    vdb.client.delete_collection(vdb.collection.name)
    vdb.__init__() # Re-initialize the collection

def get_unique_categories():
    """Get all unique purpose categories currently in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT purpose FROM documents")
    cats = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return sorted(cats)

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
