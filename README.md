# 🏥 Clinical RAG System  
*A production-ready Retrieval-Augmented Generation pipeline for clinical text using MIMIC-IV-EXT, E5-small-v2, ChromaDB, and Mistral-7B.*

This system processes large-scale clinical notes, generates embeddings, stores them in a vector database, and answers clinical questions using a grounded LLM. It includes:

- Full data pipeline (cleaning → chunking → embeddings → vector DB)
- RAG inference pipeline
- Evaluation suite (retrieval + generation)
- Streamlit UI
- Dockerized deployment

---

## 1. Architecture Diagram

```
                   ┌─────────────────────────────┐
                   │       MIMIC-IV-EXT           │
                   │   (Clinical Notes Dataset)   │
                   └───────────────┬─────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │      DataProcessor       │
                     │  - Clean text            │
                     │  - Remove PHI            │
                     │  - Chunk to 400 tokens   │
                     │  - Extract metadata      │
                     └───────────────┬──────────┘
                                     │
                                     ▼
                     ┌──────────────────────────┐
                     │   EmbeddingGenerator     │
                     │  - E5-small-v2 model     │
                     │  - Normalized vectors    │
                     │  - Store embeddings      │
                     └───────────────┬──────────┘
                                     │
                                     ▼
                     ┌──────────────────────────┐
                     │        ChromaDB          │
                     │ - HNSW cosine similarity │
                     │ - Persistent storage     │
                     └───────────────┬──────────┘
                                     │
                                     ▼
                     ┌──────────────────────────┐
                     │       RAG Pipeline       │
                     │ - Encode query (E5)      │
                     │ - Retrieve top-K chunks  │
                     │ - Format context          │
                     │ - Generate answer (7B)    │
                     └───────────────┬──────────┘
                                     │
                                     ▼
                     ┌──────────────────────────┐
                     │       Streamlit UI       │
                     │ - Query interface         │
                     │ - Show retrieval + answer │
                     └──────────────────────────┘
```

---

## 2. Dataset Instructions (MIMIC-IV-EXT)

### Dataset Location
MIMIC-IV-EXT is available on PhysioNet:  
https://physionet.org/content/mimic-iv-note/2.2/

### Steps to Access
1. Create a PhysioNet account  
2. Complete the CITI training  
3. Sign the Data Use Agreement (DUA)  
4. Download the dataset  
5. Place it here:

```
<your-path>/mimic-iv-ext-direct-1.0.0/
```

### Structure
```
mimic-iv-ext-direct-1.0.0/
    └── disease_category/
         └── subtype/
              └── file.json
```

---

## 3. Running the Pipeline

### Option A — Google Colab
```python
%run v02_claude.py
```

### Option B — Local
Install dependencies:

```bash
pip install transformers accelerate bitsandbytes sentence-transformers
pip install chromadb pandas pyarrow tqdm streamlit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Run:
```bash
python v02_claude.py
```

---

## 4. Streamlit App

Start:
```bash
streamlit run streamlit_app.py
```

---

## 5. Evaluation Metrics

### Retrieval
- Precision@K  
- Recall@K  
- NDCG  
- MRR  
- MAP  

### Generation
- ROUGE  
- BERTScore  
- Hallucination detection  

---

## 6. Configuration

Modify inside `Config`:

```python
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 100
DEFAULT_TOP_K = 5
MAX_NEW_TOKENS = 512
```

---

## 7. Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install transformers accelerate bitsandbytes sentence-transformers chromadb pandas pyarrow tqdm streamlit torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml
```yaml
version: "3.9"
services:
  clinical_rag:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models_cache:/app/models_cache
      - ./chroma_db:/app/chroma_db
      - ./processed_data:/app/processed_data
    restart: unless-stopped
```

---

## 8. Summary

This project provides:

- Full clinical RAG system  
- Chunking + embedding + vector DB  
- Mistral-7B grounded reasoning  
- Evaluation suite  
- Streamlit UI  
- Docker deployment  

Ready for clinical research and prototyping.
