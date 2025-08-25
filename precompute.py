import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from data_processing import load_data, preprocess_data

VECTORSTORE_DIR = 'vectorstore'
DATASET_PATH = 'flipkart_com-ecommerce_sample.csv'

def process_data_no_streamlit(refined_df):
    refined_df['combined_info'] = refined_df.apply(
        lambda row: f"Product ID: {row['pid']}. Product URL: {row['product_url']}. "
                    f"Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. "
                    f"Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. "
                    f"Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. "
                    f"Brand: {row['brand']}. Gender: {row['gender']}",
        axis=1
    )

    loader = DataFrameLoader(refined_df, page_content_column="combined_info")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts = []
    for i, doc in enumerate(docs):
        texts.extend(text_splitter.split_documents([doc]))
        print(f"Processed {i+1}/{len(docs)} documents")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def save_vectorstore(vectorstore, directory):
    vectorstore.save_local(directory)

# -----------------------------
# Precompute vectorstore
# -----------------------------
df = load_data(DATASET_PATH)
refined_df = preprocess_data(df)

print("Processing data and creating vectorstore...")
vectorstore = process_data_no_streamlit(refined_df)

if not os.path.exists(VECTORSTORE_DIR):
    os.makedirs(VECTORSTORE_DIR)

save_vectorstore(vectorstore, VECTORSTORE_DIR)
print("✅ Vectorstore precomputed and saved!")
