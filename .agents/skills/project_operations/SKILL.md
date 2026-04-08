---
name: project_operations
description: Core operations and contextual knowledge for the Smart Doc Finder project.
---
# Smart Doc Finder - Project Operations

Core objectives: PDF to image parsing, metadata extraction via Ollama, and searchable interaction via Streamlit.

## Project Context
**Smart Doc Finder** is an intelligent document management system that processes PDF files through a four-stage pipeline:
1. **Parsing**: Converting PDF pages into images to preserve layout and visual context.
2. **Extraction**: Using a Local LLM (Ollama) with vision capabilities to extract structured metadata from the images.
3. **Indexing**: Storing the extracted metadata and file references in a local SQLite database.
4. **Search/Retrieval**: A Streamlit-based interface for searching and viewing documents/images based on their metadata.

## Tech Stack
- **Language**: Python 3.10+
- **UI Framework**: Streamlit
- **LLM Engine**: Ollama (Local)
- **Database**: SQLite
- **Core Libraries**:
  - `pdf2image`: For PDF to image conversion (requires `poppler`).
  - `ollama`: For interacting with local vision-capable models.
  - `sqlite3`: Standard library for metadata storage.
  - `pillow`: For image handling and processing.

## Development Workflows

### 1. Environment Setup
```bash
# macOS dependencies
brew install poppler

# Python environment
python -m venv venv
source venv/bin/activate
pip install streamlit ollama pdf2image pillow
```

### 2. Pipeline Architecture
- **Stage 1 (PDF Parser)**: Logic to iterate through PDFs in a directory and save pages as images in a `data/images` folder.
- **Stage 2 (Vision Extractor)**: Prompts for Ollama to extract fields like "Total Amount", "Date", "Vendor", or general "Summary".
- **Stage 3 (Database Manager)**: SQLite schema handling document paths and JSON-formatted metadata.

### 3. Running the App
```bash
streamlit run app.py
```

## Agent Guidelines
- **Structured Extraction**: Always prompt Ollama to return JSON-formatted strings for easy parsing into the SQLite database.
- **Efficient Processing**: Implement check pointing so files aren't re-processed if they already exist in the database.
- **UI Quality**: The search interface should allow filtering by extracted fields (e.g., date ranges, vendors) and provide a side-by-side view of metadata and document images.
