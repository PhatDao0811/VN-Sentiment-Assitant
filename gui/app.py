import streamlit as st
import pandas as pd
import os
import sys

# Thêm đường dẫn core để import các modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import db_manager
# Sửa: BỎ 'ValueError' khỏi import, vì nó là một exception có sẵn
from core.nlp_engine import load_sentiment_pipeline, classify_sentiment
# ValueError sẽ được bắt trực tiếp trong handle_classification

# --- Cấu hình Trang ---
st.set_page_config(
    page_title="Trợ Lý Phân Loại Cảm Xúc Tiếng Việt (Transformer)",
    layout="wide"
)

# Khởi tạo DB khi ứng dụng bắt đầu
if not os.path.exists('data'):
    os.makedirs('data')
db_manager.init_db()

# Tải pipeline (sẽ được cache)
sentiment_pipeline = load_sentiment_pipeline()

# --- Tiêu đề & Giới thiệu ---
st.title("🇻🇳 Trợ Lý Phân Loại Cảm Xúc Tiếng Việt")
st.markdown("Sử dụng Transformer (PhoBERT) và Pipeline `sentiment-analysis`.")
st.write("---")

# --- Phần Nhập liệu & Phân loại ---
st.header("1. Nhập liệu")
input_text = st.text_area(
    "Nhập câu tiếng Việt bạn muốn phân loại:",
    placeholder="Ví dụ: Phim này hay lắm; Hôm nay tôi rất vui; Món ăn này dở quá...",
    height=100
)


# Hàm phân loại chính
def handle_classification():
    if not input_text:
        st.warning("Vui lòng nhập văn bản để phân loại.")
        return

    if sentiment_pipeline is None:
        st.error("Lỗi: Không thể tải mô hình Transformer. Vui lòng kiểm tra kết nối mạng và thư viện.")
        return

    # Dùng st.spinner thay cho threading/async
    with st.spinner('Đang phân loại cảm xúc...'):
        try:
            # Gọi hàm phân loại
            result = classify_sentiment(input_text, sentiment_pipeline)

            # Lưu lịch sử vào DB
            db_manager.save_sentiment(result['text'], result['sentiment'])

            # Hiển thị kết quả
            st.session_state['classification_result'] = result

        except ValueError as e:
            # Xử lý lỗi (vd: Câu quá ngắn) và hiển thị Pop-up thông báo lỗi
            st.error(f"⚠️ Lỗi nhập liệu: {str(e)}")
            st.session_state['classification_result'] = None
        except Exception as e:
            st.error(f"Lỗi không xác định trong quá trình phân loại: {str(e)}")
            st.session_state['classification_result'] = None


if st.button('🚀 Phân loại Cảm xúc', on_click=handle_classification):
    pass  # Xử lý đã nằm trong handle_classification

# Hiển thị kết quả sau khi phân loại
if 'classification_result' in st.session_state and st.session_state['classification_result']:
    result = st.session_state['classification_result']
    st.header("2. Kết quả Phân loại")

    sentiment = result['sentiment']
    score = result['score']

    # Gán icon/màu sắc tương ứng
    if sentiment == 'POSITIVE':
        color = 'green'
        icon = '👍'
    elif sentiment == 'NEGATIVE':
        color = 'red'
        icon = '👎'
    else:
        color = 'gray'
        icon = '😐'

    st.markdown(f"**Văn bản đã chuẩn hóa:** `{result['text']}`")
    st.markdown(f"**Nhãn Cảm xúc:** <span style='color:{color}; font-size: 24px;'>{icon} **{sentiment}**</span>",
                unsafe_allow_html=True)
    st.caption(f"Độ tin cậy: {score:.2f} (Model: PhoBERT)")

st.write("---")

# --- Phần Lịch sử Phân loại ---
st.header("3. Lịch sử Phân loại Cục bộ (SQLite)")

history_data = db_manager.get_history()

if history_data:
    # Chuyển đổi sang DataFrame để hiển thị bảng (yêu cầu báo cáo)
    df = pd.DataFrame(history_data)
    df.columns = ["Nội dung", "Cảm xúc", "Thời gian"]
    # Hiển thị 50 bản ghi mới nhất
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Chưa có lịch sử phân loại nào được lưu.")

# Chạy ứng dụng bằng lệnh: streamlit run gui/app.py