import sqlite3
import os
import datetime

DB_PATH = "events.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Active live logs table (shown in UI feed)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT
        )
    ''')
    # Master permanent archive table (stores all historical proofs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id INTEGER,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT,
            archived_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_event(event_type, description, image_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (timestamp, event_type, description, image_path) VALUES (?, ?, ?, ?)",
        (timestamp, event_type, description, image_path)
    )
    conn.commit()
    conn.close()

def get_recent_logs(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def archive_and_clear_logs(proofs_dir="security_proofs"):
    """
    Archives active logs to:
    1. Permanent SQLite table `logs_archive`
    2. Timestamped formatted text audit file in `security_proofs/`
    Then cleans the active `logs` table so the UI looks clean.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp ASC")
    active_logs = cursor.fetchall()

    if not active_logs:
        conn.close()
        return False, None, 0

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Archive to logs_archive table
    for log in active_logs:
        log_id, ts, event_type, desc, img_path = log
        cursor.execute(
            "INSERT INTO logs_archive (original_id, timestamp, event_type, description, image_path, archived_at) VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, ts, event_type, desc, img_path, now_str)
        )

    # 2. Archive to formatted text proof file
    if not os.path.exists(proofs_dir):
        os.makedirs(proofs_dir)

    proof_file_path = os.path.join(proofs_dir, f"security_proof_{file_timestamp}.txt")
    with open(proof_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("          SMART AI SURVEILLANCE - SECURITY AUDIT PROOF\n")
        f.write("=" * 70 + "\n")
        f.write(f"Archived Date & Time : {now_str}\n")
        f.write(f"Total Incidents Logged: {len(active_logs)}\n")
        f.write("=" * 70 + "\n\n")
        for log in active_logs:
            log_id, ts, event_type, desc, img_path = log
            f.write(f"[{ts}] [{event_type}] {desc}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Audit Proof Report\n")
        f.write("=" * 70 + "\n")

    # 3. Clear active screen logs table
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

    return True, proof_file_path, len(active_logs)

def clear_logs():
    archive_and_clear_logs()


# Auto-initialize DB on import
init_db()

if __name__ == "__main__":
    print("Database initialized successfully.")
