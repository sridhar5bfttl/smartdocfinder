import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import json
from database import get_all_documents, init_db, insert_document, get_documents_by_ids, search_documents, clear_all_documents, get_unique_categories, reindex_all_for_vectors, delete_document, cleanup_missing_documents, update_document_metadata
from parser import PDFParser
from vector_db import VectorDB
import pandas as pd
from extractor import AIExtractor
import time

# Set up page config
st.set_page_config(
    page_title="Smart Doc Finder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States for Stop functionality
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'draft_images' not in st.session_state:
    st.session_state.draft_images = []
if 'selected_pages' not in st.session_state:
    st.session_state.selected_pages = {} # map image_path -> boolean
if 'editing_doc' not in st.session_state:
    st.session_state.editing_doc = None

# Ensure database is initialized
init_db()

# Auto-Sync Vector DB on startup if it's empty but SQLite has data
try:
    sqlite_docs = get_all_documents()
    vdb = VectorDB()
    # Check if we have data in SQLite but none in Vectors
    if len(sqlite_docs) > 0 and vdb.collection.count() == 0:
        reindex_all_for_vectors()
except Exception as e:
    # Silent fail for startup sync
    pass

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# Application Title
st.title("📄 Smart Doc Finder")

# Show cancellation message if a stop was requested
if st.session_state.stop_requested:
    st.warning("⚠️ Processing cancelled by user. Partial results have been saved to the database.")
    st.session_state.stop_requested = False

st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=80)
    st.header("Control Panel")
    
    st.markdown("### 🧠 Vision Engine")
    vision_model = st.selectbox(
        "Select Model",
        ["moondream", "llama3.2-vision"],
        index=0,
        help="Moondream (1.6B) is extremely fast. Llama 3.2-Vision (11B) is more accurate but slower."
    )

    st.markdown("---")
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")
    
    if uploaded_file:
        # Save uploaded file to the raw directory
        raw_path = os.path.join("data/raw", uploaded_file.name)
        os.makedirs("data/raw", exist_ok=True)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # --- PHASE 1: PREVIEW & SELECTION ---
        if st.button("🖼️ Step 1: Generate Previews", use_container_width=True):
            with st.spinner("Converting PDF pages to images..."):
                parser = PDFParser()
                st.session_state.draft_images = parser.convert_pdf_to_images(raw_path)
                # Default all to selected initially
                st.session_state.selected_pages = {img: True for img in st.session_state.draft_images}
                st.rerun()

        # --- PHASE 2: AI EXTRACTION ---
        # Only show the "Process Selected" button if we have previews and aren't already processing
        if st.session_state.draft_images and not st.session_state.is_processing:
            if st.button("🚀 Step 2: Index Selected with AI", use_container_width=True):
                st.session_state.is_processing = True
                st.session_state.stop_requested = False
                st.rerun()
        
        # Stop Button (shows during AI phase)
        if st.session_state.is_processing:
            if st.button("🛑 Stop Processing", type="primary", use_container_width=True):
                st.session_state.stop_requested = True
                st.session_state.is_processing = False
                st.rerun()

    st.markdown("---")
    st.markdown("---")
    st.header("🔍 Filters")
    min_amt = st.number_input("Min Amount ($)", min_value=0.0, value=0.0, step=10.0)
    max_amt = st.number_input("Max Amount ($)", min_value=0.0, value=10000.0, step=10.0)
    
    # Dynamic categories from DB
    available_cats = get_unique_categories()
    if not available_cats:
        # Fallback if DB is empty
        available_cats = ["Vehicle", "Tuition", "Medical", "Business", "Personal", "Other"]
        
    selected_cats = st.multiselect("Categories", available_cats, default=available_cats)
    
    st.markdown("---")
    st.header("⚙️ Maintenance")
    if st.button("🗑️ Clear All Data", help="Wipe SQLite and Vector DB", use_container_width=True):
        clear_all_documents()
        st.success("Wiped all data!")
        st.rerun()
        
    if st.button("🔄 Re-index Vectors", help="Sync existing documents to Semantic Search index", use_container_width=True):
        with st.spinner("Re-building semantic index..."):
            count = reindex_all_for_vectors()
            st.success(f"Successfully re-indexed {count} pages!")
            st.rerun()

    if st.button("🧹 Auto-Cleanup", help="Purge documents with missing images", use_container_width=True):
        count = cleanup_missing_documents()
        st.success(f"Cleaned up {count} broken records!")
        st.rerun()

    st.markdown("---")
    st.header("🔬 Debug")
    show_diagnostics = st.toggle("Show AI Diagnostics", value=False, help="Show semantic similarity scores and raw database IDs.")

# Main Search and Results Area
st.header("🔍 Document Search")

# --- GRID SELECTION UI ---
if st.session_state.draft_images and not st.session_state.is_processing:
    with st.expander("🖼️ Page Selection - Review pages before indexing", expanded=True):
        st.info("Select the pages you want the AI to index into your database.")
        cols_count = 4
        cols = st.columns(cols_count)
        for i, img_path in enumerate(st.session_state.draft_images):
            with cols[i % cols_count]:
                st.image(img_path, use_container_width=True)
                # Update selection state
                is_selected = st.checkbox(f"Page {i+1}", value=st.session_state.selected_pages.get(img_path, True), key=f"sel_{i}")
                st.session_state.selected_pages[img_path] = is_selected

# --- AI PROCESSING EXECUTION ---
if st.session_state.is_processing:
    # Identify which pages were selected
    selected_to_process = [img for img, sel in st.session_state.selected_pages.items() if sel]
    
    if not selected_to_process:
        st.warning("⚠️ No pages selected! Please check at least one page in the selection grid above.")
        st.session_state.is_processing = False
    else:
        with st.status("🧠 AI is processing your selected pages...", expanded=True) as status:
            extractor = AIExtractor(model=vision_model)
            for i, img_path in enumerate(selected_to_process):
                # Check for manual stop
                if st.session_state.stop_requested:
                    break
                    
                page_start = time.time()
                st.markdown(f"#### 📄 AI reading: {os.path.basename(img_path)}")
                st.image(img_path, width=300)
                
                info_p = st.empty()
                info_p.info(f"⏳ Running {vision_model} extraction...")
                
                metadata = extractor.extract_metadata(img_path)
                # Use a dummy page number or infer from filename if needed, 
                # but for now we just use the loop index + 1 or better, find it from draft_images
                original_page_num = st.session_state.draft_images.index(img_path) + 1
                
                insert_document(uploaded_file.name, img_path, original_page_num, metadata)
                
                elapsed = time.time() - page_start
                info_p.success(f"✅ Indexed in {elapsed:.1f}s")
                st.divider()

            if st.session_state.stop_requested:
                status.update(label="🛑 Processing Cancelled", state="error", expanded=False)
                st.session_state.is_processing = False
            else:
                status.update(label="✅ Selective Indexing Complete!", state="complete", expanded=False)
                st.session_state.is_processing = False
                st.session_state.draft_images = [] # Clear drafts after successful indexing
                st.balloons()
                st.rerun()

search_mode = st.radio("Search Mode", ["Standard (Keyword)", "Smart (Semantic)"], horizontal=True)
search_query = st.text_input("Search query...", placeholder="e.g. 'medical bill' or 'tuition fee'")

# Fetch and filter documents
if search_mode == "Standard (Keyword)":
    # Use optimized SQL search with filters
    docs = search_documents(
        query=search_query if search_query else None,
        min_amount=min_amt,
        max_amount=max_amt,
        categories=selected_cats
    )
else:
    # Smart Search via Vector DB
    if search_query:
        with st.spinner("Searching semantically..."):
            try:
                vdb = VectorDB()
                results = vdb.semantic_search(search_query, n_results=15)
                # Store scores for diagnostics
                scores_map = {str(r['id']): r['distance'] for r in results}
                
                doc_ids = [r['id'] for r in results]
                raw_docs = get_documents_by_ids(doc_ids)
                
                # In Smart Mode, we prioritize showing the AI matches over sidebar filters
                # This ensures the user NEVER sees "No documents found" when matches exist
                docs = raw_docs
            except Exception as e:
                st.error(f"Semantic search failed: {e}")
                docs = []
    else:
        # If no query but Smart mode is on, just show all filtered docs
        docs = search_documents(min_amount=min_amt, max_amount=max_amt, categories=selected_cats)
        scores_map = {}

if docs:
    st.markdown(f"Found **{len(docs)}** results")
    
    # CSV Export Button
    try:
        df = pd.DataFrame(docs)
        # Clean up for CSV (drop columns or reorder)
        export_df = df[['original_pdf', 'page_number', 'doc_date', 'amount', 'vendor', 'purpose', 'doc_type', 'created_at']]
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export These Results to CSV",
            data=csv_data,
            file_name="smart_doc_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Export feature temporarily unavailable: {e}")
    
    for doc in docs:
        with st.container():
            # Show diagnostic info if enabled
            if show_diagnostics:
                score = scores_map.get(str(doc['id']), "N/A")
                if isinstance(score, (float, int)):
                    accuracy = max(0, 100 - (score * 100))
                    st.caption(f"🆔 ID: {doc['id']} | 🎯 Semantic Accuracy: {accuracy:.1f}%")
                else:
                    st.caption(f"🆔 ID: {doc['id']} | 🎯 Score: {score}")

            st.markdown(f'<div class="stCard">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show the page image
                if os.path.exists(doc['image_path']):
                    st.image(doc['image_path'], caption=f"Page {doc['page_number']}", use_container_width=True)
                else:
                    st.error("Image file missing")
            
            with col2:
                st.subheader(f"📄 {doc['original_pdf']}")
                
                # Display metadata in a neat layout
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.write(f"**📅 Date:** {doc['doc_date'] or 'N/A'}")
                    st.write(f"**🏢 Vendor:** {doc['vendor'] or 'N/A'}")
                with m_col2:
                    st.write(f"**💰 Amount:** ${doc['amount'] or 0.0:,.2f}")
                    st.write(f"**🎯 Purpose:** {doc['purpose'] or 'N/A'}")
                
                st.write(f"**🏷️ Doc Type:** {doc['doc_type'] or 'Unknown'}")
                
                # Metadata Expander and Delete button
                c_1, c_2, c_3 = st.columns([2, 1, 1])
                with c_1:
                    with st.expander("🛠️ Details"):
                        st.code(f"Path: {doc['image_path']}")
                        try:
                            meta_data = json.loads(doc['metadata_json'])
                            st.json(meta_data)
                        except:
                            st.write("Raw:", doc['metadata_json'])
                with c_2:
                    if st.button("✏️ Edit", key=f"edit_{doc['id']}", use_container_width=True):
                        st.session_state.editing_doc = doc['id']
                        st.rerun()
                with c_3:
                    if st.button("🗑️ Delete", key=f"del_{doc['id']}", type="secondary", use_container_width=True):
                        delete_document(doc['id'])
                        st.rerun()
                
                # Manual Correction Form
                if st.session_state.editing_doc == doc['id']:
                    st.markdown("---")
                    with st.form(key=f"form_{doc['id']}"):
                        st.subheader("📝 Manual Correction")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            new_vendor = st.text_input("Vendor", value=doc['vendor'] or "")
                            new_date = st.text_input("Date (YYYY-MM-DD)", value=doc['doc_date'] or "")
                        with col_b:
                            new_amount = st.number_input("Amount ($)", value=float(doc['amount'] or 0.0), step=0.01)
                            # Use existing categorized purpose list
                            try:
                                p_idx = available_cats.index(doc['purpose']) if doc['purpose'] in available_cats else 0
                            except:
                                p_idx = 0
                            selected_purpose = st.selectbox("Pick Existing Category", available_cats, index=p_idx)
                            custom_purpose = st.text_input("...or type a NEW Category", placeholder="e.g. Taxes, Insurance")
                        
                        f_btn1, f_btn2 = st.columns(2)
                        with f_btn1:
                            if st.form_submit_button("✅ Save Changes", use_container_width=True):
                                # Prioritize custom purpose if provided
                                final_purpose = custom_purpose.strip() if custom_purpose.strip() else selected_purpose
                                
                                updated_meta = {
                                    "vendor": new_vendor,
                                    "date": new_date,
                                    "amount": new_amount,
                                    "purpose": final_purpose,
                                    "doc_type": doc['doc_type'] or "document"
                                }
                                update_document_metadata(doc['id'], updated_meta)
                                st.session_state.editing_doc = None
                                st.success("Document updated!")
                                st.rerun()
                        with f_btn2:
                            if st.form_submit_button("❌ Cancel", use_container_width=True):
                                st.session_state.editing_doc = None
                                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No documents found. Try a different search or upload a PDF document in the sidebar to process it.")

# Footer
st.markdown("---")
st.caption("Powered by Smart Doc Finder Engine | Phase 1 Scaffolding")
