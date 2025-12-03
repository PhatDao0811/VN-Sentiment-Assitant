import sqlite3
import datetime

DATABASE_NAME = "data/sentiment_history.db"


def init_db():
    """Khởi tạo database và tạo bảng sentiments nếu chưa tồn tại."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Lỗi khi khởi tạo database: {e}")
    finally:
        if conn:
            conn.close()


def save_sentiment(text: str, sentiment: str):
    """Lưu kết quả phân loại vào database (Sử dụng parameterized queries)."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Parameterized query để tránh SQL injection
        cursor.execute(
            "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
            (text, sentiment, current_time)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")
    finally:
        if conn:
            conn.close()


def get_history(limit: int = 50):
    """Lấy lịch sử phân loại mới nhất."""
    conn = None
    results = []
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Truy vấn giới hạn 50 bản ghi mới nhất
        cursor.execute(
            "SELECT text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        results = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Lỗi khi truy vấn lịch sử: {e}")
    finally:
        if conn:
            conn.close()

    # Chuyển đổi sang list of dict để dễ dùng hơn
    history_list = [{"text": row[0], "sentiment": row[1], "timestamp": row[2]} for row in results]
    return history_list


if __name__ == '__main__':
    # Tạo folder data và khởi tạo DB cho lần chạy đầu tiên
    import os

    if not os.path.exists('data'):
        os.makedirs('data')
    init_db()
    print("Database đã được khởi tạo/kiểm tra.")