# Offline PDF RAG Chatbot

Upload any PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).
This app works fully offline with local models via Ollama.


## Features

* Upload PDFs (up to 200MB)
* Extract and embed text using FAISS Vector Store
* Query the PDF with local LLMs (e.g., llama3)
* Simple and interactive Streamlit UI
* Works offline — no external API required.

## Tech Stack

* **Python 3.11**
* **LangChain** – RAG pipeline (embeddings + retriever)
* **FAISS** – Vector database for embeddings
* **Ollama** – Local LLM inference (llama3)
* **Streamlit** – Web UI
* **PyPDF** – PDF text extraction

## Architecture

![Architecture Diagram](/architecture.png)


## Installation & Setup

1. Clone the repo

```
git clone https://github.com/iamakashjha/RAG-based-chatbot.git
cd RAG-based-chatbot
```

2. Create a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Install Ollama (for macOS/Linux)

* Download [Ollama](/https://ollama.com/download?utm_source=chatgpt.com)
* Pull your preferred model:

```
ollama pull llama3
```

## Running the App

```
streamlit run app.py
```

## Author

[Akash Jha - LinkedIn](/https://www.linkedin.com/in/iamakashjha1/)

