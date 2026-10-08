import os
import sqlite3
from typing import Optional

# Proje dizin yollarını dinamik olarak belirle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")
ANOMALIES_DIR = os.path.join(EXPORTS_DIR, "anomalies")
DB_PATH = os.path.join(EXPORTS_DIR, "flight_telemetry.db")


def init_db():
    """Gerekli klasörleri oluşturur ve SQLite tablolarını ayağa kaldırır."""
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    os.makedirs(ANOMALIES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Anlık Telemetri Kayıtları Tablosu
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS telemetry_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        roll_angle REAL NOT NULL,
        pitch_offset REAL NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL
    );
    """)

    # 2. Kritik Durum / Anomali Olayları Tablosu
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flight_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        event_type TEXT NOT NULL,
        measured_value REAL,
        image_path TEXT
    );
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Veri tabanı hazırlandı: {DB_PATH}")


def log_telemetry(
    frame_id: int,
    timestamp: float,
    roll_angle: float,
    pitch_offset: float,
    confidence: float,
    status: str,
):
    """Görüntü işleme motorundan gelen anlık kare verilerini kaydeder."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    INSERT INTO telemetry_logs (frame_id, timestamp, roll_angle, pitch_offset, confidence, status)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        (frame_id, timestamp, roll_angle, pitch_offset, confidence, status),
    )
    conn.commit()
    conn.close()


def log_event(
    frame_id: int,
    timestamp: float,
    event_type: str,
    measured_value: Optional[float] = None,
    image_path: Optional[str] = None,
):
    """Kritik yatış açısı ve ufuk kaybı anormalliklerini kaydeder."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    INSERT INTO flight_events (frame_id, timestamp, event_type, measured_value, image_path)
    VALUES (?, ?, ?, ?, ?)
    """,
        (frame_id, timestamp, event_type, measured_value, image_path),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()