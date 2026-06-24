# Kenyan Legal RAG 2025

This project provides a specialized Retrieval-Augmented Generation (RAG) interface for querying Kenyan High Court judgments and specialized legal precedents.

## Features
- **PDF Processing**: Upload and index multiple High Court judgments.
- **Legal Context**: Specifically tuned for the Kenyan legal framework and Constitution 2010.
- **Vector Storage**: Uses ChromaDB for persistent storage of legal embeddings.
- **Citation Focus**: Encourages the model to cite Case Names and Petition Numbers.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run legal_bot.py`
3. Provide your OpenAI API Key in the sidebar.
4. Upload Kenyan High Court PDFs (sourced from Kenya Law Reports/e-citizen).

## Disclaimer
This tool is for educational and research assistance only. It does not constitute legal advice. Users should consult registered advocates of the High Court of Kenya for professional legal matters.