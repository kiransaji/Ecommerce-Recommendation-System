# import os
# import time
# from dotenv import load_dotenv
# import streamlit as st
# from langchain.document_loaders import DataFrameLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from transformers import pipeline

# # Load environment variables
# load_dotenv()

# # -------------------------
# # Helper functions
# # -------------------------

# def save_vectorstore(vectorstore, directory):
#     """Save the FAISS vectorstore to disk."""
#     vectorstore.save_local(directory)

# def load_vectorstore(directory, embeddings):
#     """Load the FAISS vectorstore from disk."""
#     return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)

# def process_data(refined_df):
#     """Process the DataFrame and create a FAISS vectorstore with embeddings."""
#     # Combine product info into one text column
#     refined_df['combined_info'] = refined_df.apply(
#         lambda row: f"Product ID: {row['pid']}. Product URL: {row['product_url']}. "
#                     f"Product Name: {row['product_name']}. Primary Category: {row['primary_category']}. "
#                     f"Retail Price: ${row['retail_price']}. Discounted Price: ${row['discounted_price']}. "
#                     f"Primary Image Link: {row['primary_image_link']}. Description: {row['description']}. "
#                     f"Brand: {row['brand']}. Gender: {row['gender']}",
#         axis=1
#     )

#     loader = DataFrameLoader(refined_df, page_content_column="combined_info")
#     docs = loader.load()

#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
#     # Split docs and show progress
#     texts = []
#     total_docs = len(docs)
#     progress_bar = st.progress(0)
#     for i, doc in enumerate(docs):
#         chunked = text_splitter.split_documents([doc])
#         texts.extend(chunked)
#         progress = int((i + 1) / total_docs * 100)
#         progress_bar.progress(progress)
#         time.sleep(0.01)

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(texts, embeddings)

#     progress_bar.progress(100)
#     st.success("✅ Vectorstore created!")
#     return vectorstore

# # -------------------------
# # Main Streamlit function
# # -------------------------

# def run_recommendation_app(refined_df):
#     st.header("Product Recommendation")

#     vectorstore_dir = 'vectorstore'
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Load or create vectorstore
#     if os.path.exists(vectorstore_dir):
#         vectorstore = load_vectorstore(vectorstore_dir, embeddings)
#     else:
#         st.info("Vectorstore not found. Creating vectorstore, please wait...")
#         vectorstore = process_data(refined_df)
#         save_vectorstore(vectorstore, vectorstore_dir)

#     # --- LLM pipeline ---
#     @st.cache_resource
#     def get_llm_pipeline():
#         return pipeline(
#             "text2text-generation",
#             model="google/flan-t5-small",
#             max_new_tokens=128,
#             temperature=0.5
#         )

#     llm_pipeline = get_llm_pipeline()

#     # --- User inputs ---
#     department = st.text_input("Product Department")
#     category = st.text_input("Product Category")
#     brand = st.text_input("Product Brand")
#     price = st.text_input("Maximum Price Range")
#     top_k = st.slider("Number of similar products to consider", 1, 10, 3)

#     if st.button("Get Recommendations"):
#         with st.spinner("Searching similar products..."):
#             # Build a search query
#             query_text = f"Department: {department}, Category: {category}, Brand: {brand}, Max Price: {price}"

#             # Perform FAISS similarity search
#             similar_docs = vectorstore.similarity_search(query_text, k=top_k)
#             context_text = "\n".join([doc.page_content for doc in similar_docs])

#         with st.spinner("Generating refined recommendations..."):
#             # LLM prompt with context
#             prompt = f"""
#             Based on the following similar products, suggest three recommended products that best match the user's input:

#             User Input:
#             Department: {department}
#             Category: {category}
#             Brand: {brand}
#             Maximum Price: {price}

#             Similar Products:
#             {context_text}
#             """

#             result = llm_pipeline(prompt)
#             recommendations = result[0]['generated_text']

#             st.success("✅ Recommendations ready!")
#             st.write(recommendations)

import os
import time
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import pipeline

# Load environment variables
load_dotenv()

# -------------------------
# Helper functions
# -------------------------

def save_vectorstore(vectorstore, directory):
    """Save the FAISS vectorstore to disk."""
    vectorstore.save_local(directory)

def load_vectorstore(directory, embeddings):
    """Load the FAISS vectorstore from disk."""
    return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)

def process_data(refined_df):
    """Process the DataFrame and create a FAISS vectorstore with embeddings."""
    # Combine product info into one text column
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
    
    # Split docs and show progress
    texts = []
    total_docs = len(docs)
    progress_bar = st.progress(0)
    for i, doc in enumerate(docs):
        chunked = text_splitter.split_documents([doc])
        texts.extend(chunked)
        progress = int((i + 1) / total_docs * 100)
        progress_bar.progress(progress)
        time.sleep(0.01)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    progress_bar.progress(100)
    st.success("✅ Vectorstore created!")
    return vectorstore

# -------------------------
# Main Streamlit function
# -------------------------

def run_recommendation_app(refined_df):
    st.header("Product Recommendation")

    vectorstore_dir = 'vectorstore'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load or create vectorstore
    if os.path.exists(vectorstore_dir):
        vectorstore = load_vectorstore(vectorstore_dir, embeddings)
    else:
        st.info("Vectorstore not found. Creating vectorstore, please wait...")
        vectorstore = process_data(refined_df)
        save_vectorstore(vectorstore, vectorstore_dir)

    # --- LLM pipeline ---
    @st.cache_resource
    def get_llm_pipeline():
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=256,
            temperature=0.5
        )

    llm_pipeline = get_llm_pipeline()

    # --- User inputs ---
    department = st.text_input("Product Department")
    category = st.text_input("Product Category")
    brand = st.text_input("Product Brand")
    price = st.text_input("Maximum Price Range")
    top_k = st.slider("Number of similar products to consider", 1, 10, 3)

    if st.button("Get Recommendations"):
        with st.spinner("Searching similar products..."):
            query_text = f"Department: {department}, Category: {category}, Brand: {brand}, Max Price: {price}"
            similar_docs = vectorstore.similarity_search(query_text, k=top_k)

        # --- Display similar products ---
        st.subheader("🔹 Similar Products Found")
        for doc in similar_docs:
            content = doc.page_content
            lines = content.split(". ")
            name_line = lines[2] if len(lines) > 2 else ""
            url_line = lines[1] if len(lines) > 1 else ""
            img_line = lines[6] if len(lines) > 6 else ""

            st.markdown(f"**{name_line}**")
            st.markdown(f"[View Product]({url_line.split(': ')[-1]})")
            st.image(img_line.split(': ')[-1], width=200)
            st.markdown("---")

        # --- LLM generation ---
        with st.spinner("Generating refined recommendations..."):
            context_text = "\n".join([doc.page_content for doc in similar_docs])

            prompt = f"""
            Based on the following similar products, suggest three recommended products in the format:
            Product Name: <name>, Product URL: <url>, Primary Image Link: <image_url>

            User Input:
            Department: {department}
            Category: {category}
            Brand: {brand}
            Maximum Price: {price}

            Similar Products:
            {context_text}
            """

            result = llm_pipeline(prompt)
            raw_recommendations = result[0]['generated_text']

        # --- Parse LLM output into structured recommendations ---
        st.subheader("✅ Refined Recommendations")
        product_pattern = r"Product Name:\s*(.*?),\s*Product URL:\s*(.*?),\s*Primary Image Link:\s*(.*?)(?:$|\n)"
        matches = re.findall(product_pattern, raw_recommendations, re.MULTILINE)

        if matches:
            for name, url, img in matches:
                st.markdown(f"**{name.strip()}**")
                st.markdown(f"[View Product]({url.strip()})")
                st.image(img.strip(), width=200)
                st.markdown("---")
        else:
            # fallback if regex fails
            st.write(raw_recommendations)
