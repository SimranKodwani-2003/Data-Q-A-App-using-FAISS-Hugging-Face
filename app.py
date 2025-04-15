import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# ------------------------------
# 1. Load Embedding and QA Models
# ------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embed_model, qa_pipeline

embed_model, qa_pipeline = load_models()

# ------------------------------
# 2. UI Layout
# ------------------------------
st.title("📊 Data Q&A App using FAISS + Hugging Face")
st.markdown("Upload your dataset and ask any question related to it.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # ------------------------------
    # 3. Read and Preprocess Dataset
    # ------------------------------
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:")
    st.dataframe(df.head())

    # Convert each row into a string chunk
    data_chunks = df.astype(str).apply(lambda row: ", ".join(row.values), axis=1).tolist()

    # ------------------------------
    # 4. Generate Embeddings and Build FAISS Index
    # ------------------------------
    with st.spinner("Generating embeddings and building index..."):
        embeddings = embed_model.encode(data_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(np.array(embeddings))

    # ------------------------------
    # 5. User Query Input
    # ------------------------------
    query = st.text_input("Ask a question about your dataset:")

    if query:
        with st.spinner("Searching and answering..."):
            query_embedding = embed_model.encode([query])
            D, I = faiss_index.search(np.array(query_embedding), k=5)

            # Get top relevant chunks
            top_chunks = [data_chunks[i] for i in I[0]]
            context = " ".join(top_chunks)

            # ------------------------------
            # 6. Answer with QA Pipeline
            # ------------------------------
            answer = qa_pipeline(question=query, context=context)

            st.subheader("Answer:")
            st.write(answer['answer'])

            with st.expander("🔍 Top relevant data"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")

