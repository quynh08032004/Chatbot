
import os
import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chatbot Du Lịch Hà Nội", layout="wide")
st.title("Chatbot Du Lịch Hà Nội")
st.write("Hỏi tôi về **địa điểm**, **ẩm thực**, hoặc **trải nghiệm** ở Hà Nội nhé!")

@st.cache_resource()
def load_chatbot():
    data_path = os.path.join(os.getcwd(), "hn_tour_cuisine_dataset_seed.csv")
    vector_db_path = os.path.join(os.getcwd(), "vectorstores")
    data = pd.read_csv(data_path).fillna("")

    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    if os.path.exists(vector_db_path):
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        st.info("Đang tạo FAISS index mới (chỉ lần đầu)...")
        texts = data["question_vi"].tolist()
        db = FAISS.from_texts(texts, embedding_model)
        db.save_local(vector_db_path)

    # Lấy toàn bộ embedding để so khớp nhanh hơn
    question_embeddings = db.index.reconstruct_n(0, db.index.ntotal)
    model_path = "models/vinallama-7b-chat_q5_0.gguf"
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        temperature=0.01,
        threads=4,
        max_new_tokens=1024,
    )

    return data, embedding_model, np.array(question_embeddings), llm, db

data, embedder, question_embeddings, llm, db = load_chatbot()
query = st.text_input("Nhập câu hỏi của bạn:")

if query:
    with st.spinner("Đang suy nghĩ..."):
        try:
            # --- Tính embedding cho câu hỏi người dùng ---
            query_emb = embedder.embed_query(query)

            # --- Tính độ tương đồng cosine ---
            similarities = cosine_similarity([query_emb], question_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            threshold = 0.6  # Ngưỡng chấp nhận câu hỏi gần

            if best_score >= threshold:
                matched_question = data.loc[best_idx, "question_vi"]
                original_answer = data.loc[best_idx, "answer_vi"]

                prompt = f"""
                Bạn là chatbot du lịch Hà Nội thân thiện và hữu ích.
                Câu hỏi gốc: {matched_question}
                Câu trả lời chính xác: {original_answer}

                Hãy viết lại câu trả lời trên sao cho tự nhiên, dễ hiểu.
                Không được bịa thêm thông tin,không thêm thông tin khác.
                Trả lời:
                """
                answer = llm(prompt).strip()

                st.success(answer)
                st.caption(f" Khớp với: {matched_question} (score={best_score:.2f})")

            else:
                st.warning("Xin lỗi, tôi chưa có thông tin về câu hỏi này.")

        except Exception as e:
            st.error(f"Lỗi khi tạo câu trả lời: {e}")
