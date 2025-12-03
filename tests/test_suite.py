import sys
import os
import pandas as pd

# Thêm đường dẫn core để import nlp_engine
# Đảm bảo bạn chạy file này từ thư mục gốc của project (VN-Sentiment-Assitant)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.nlp_engine import load_sentiment_pipeline, classify_sentiment
# --- 1. Bộ Test Case Chuẩn (10 Câu) ---
# Dựa trên Mục VIII. BỘ TEST CASE (10 CÂU) trong đề tài
TEST_CASES = [
    {"input": "Hôm nay tôi rất vui", "expected_sentiment": "POSITIVE"},
    {"input": "Món ăn này dở quá", "expected_sentiment": "NEGATIVE"},
    {"input": "Thời tiết bình thường", "expected_sentiment": "NEUTRAL"},
    {"input": "Rât vui hom nay", "expected_sentiment": "POSITIVE"},  # Biến thể thiếu dấu (Rât)
    {"input": "Công việc ôn định", "expected_sentiment": "NEUTRAL"},  # Biến thể thiếu dấu (ôn)

    {"input": "Phim này hay lắm", "expected_sentiment": "POSITIVE"},
    {"input": "Tôi buồn vì thất bại", "expected_sentiment": "NEGATIVE"},
    {"input": "Ngày mai đi học", "expected_sentiment": "NEUTRAL"},
    {"input": "Cảm ơn bạn rât nhiều", "expected_sentiment": "POSITIVE"},  # Biến thể thiếu dấu (rât)
    {"input": "Mệt mỏi quá hôm nay", "expected_sentiment": "NEGATIVE"},
]


def run_tests():
    """Chạy toàn bộ 10 test cases và tính toán độ chính xác."""
    print("--- 🔬 BẮT ĐẦU CHẠY BỘ TEST CASE 10 CÂU (Yêu cầu đề tài) 🔬 ---")

    try:
        # 1. Tải Pipeline (sẽ được cache)
        sentiment_pipeline = load_sentiment_pipeline()
        if sentiment_pipeline is None:
            print("\n❌ LỖI: Không thể tải mô hình Transformer. Không thể chạy test.")
            return

        correct_predictions = 0
        results = []

        # 2. Thực hiện Test Case
        for i, case in enumerate(TEST_CASES):
            raw_text = case["input"]
            expected = case["expected_sentiment"]
            actual = ""
            status = "FAIL"

            try:
                # Gọi hàm phân loại chính
                classification_result = classify_sentiment(raw_text, sentiment_pipeline)
                actual = classification_result["sentiment"]

                # So sánh kết quả
                if actual == expected:
                    status = "PASS"
                    correct_predictions += 1

            except ValueError as e:
                actual = f"LỖI INPUT: {str(e)}"
                status = "FAIL"
            except Exception as e:
                actual = f"LỖI PHÂN LOẠI: {str(e)}"
                status = "FAIL"

            results.append({
                "STT": i + 1,
                "Đầu vào": raw_text,
                "Mong đợi": expected,
                "Thực tế": actual,
                "Trạng thái": status
            })

        # 3. Tính toán Độ chính xác
        total_cases = len(TEST_CASES)
        accuracy = (correct_predictions / total_cases) * 100

        # 4. Hiển thị kết quả (Dùng DataFrame để dễ dàng copy vào Báo cáo)
        df = pd.DataFrame(results)
        print("\n--- BẢNG KẾT QUẢ TEST CASE ---")
        print(df.to_markdown(index=False))

        # 5. Đánh giá Yêu cầu
        print("\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---")
        print(f"Tổng số case: {total_cases}")
        print(f"Số case đúng: {correct_predictions}")
        print(f"✅ Độ chính xác (Accuracy): {accuracy:.2f}%")

        if accuracy >= 65:
            print(f"🎉 ĐẠT yêu cầu đề tài (≥ 65%): {accuracy:.2f}%")
        else:
            print(f"❌ CHƯA ĐẠT yêu cầu đề tài (Cần ≥ 65%): {accuracy:.2f}%")

    except Exception as e:
        print(f"\n❌ LỖI NGHIÊM TRỌNG TRONG HỆ THỐNG TEST: {e}")


if __name__ == '__main__':
    run_tests()