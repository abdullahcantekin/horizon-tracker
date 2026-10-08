import os
import sqlite3
import time
import io
import csv
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

# Görüntü işleme ve HUD modüllerini bağlıyoruz
from core.processor import HorizonProcessor
from core.hud import HUDDisplay
from backend.database import log_telemetry, log_event

app = FastAPI(title="TUSAS Horizon Tracker Telemetry API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "exports", "flight_telemetry.db")
ANOMALIES_DIR = os.path.join(BASE_DIR, "exports", "anomalies")

# Anomali klasörünü garantiye al ve tarayıcı erişimi için static olarak bağla
os.makedirs(ANOMALIES_DIR, exist_ok=True)
app.mount("/anomalies", StaticFiles(directory=ANOMALIES_DIR), name="anomalies")

# Global video yakalama ve işleme nesneleri
processor = HorizonProcessor()
hud = HUDDisplay()

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Veri tabanı bulunamadı.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
def root():
    return {"status": "online", "system": "Horizon Tracker Telemetry Server"}

@app.get("/telemetry/latest")
def get_latest_telemetry():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"message": "Henüz veri yok."}
    return dict(row)

@app.get("/events")
def get_flight_events():
    """Kaydedilen tüm anomali ve kritik olayları döndürür."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flight_events ORDER BY event_id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/mission/debrief")
def get_mission_debrief():
    """Uçuş sonrası analitik brifing ve görev başarı raporu çıktısı üretir."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), MAX(timestamp), MAX(ABS(roll_angle)), AVG(confidence) FROM telemetry_logs")
    total_frames, max_time, max_roll, avg_conf = cursor.fetchone()

    if not total_frames or total_frames == 0:
        conn.close()
        return {"status": "NO_DATA", "message": "Henüz telemetri kaydı yok."}

    cursor.execute("SELECT COUNT(*) FROM telemetry_logs WHERE status = 'STABLE'")
    stable_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM telemetry_logs WHERE status = 'WARNING'")
    warning_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM telemetry_logs WHERE status = 'LOST'")
    lost_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM flight_events")
    total_events = cursor.fetchone()[0]

    conn.close()

    stability_rate = round((stable_count / total_frames) * 100, 2)
    mission_score = round(max(0, 100 - ((warning_count * 0.1) + (lost_count * 0.5))), 1)

    return {
        "total_frames_logged": total_frames,
        "flight_duration_sec": round(max_time or 0.0, 2),
        "peak_roll_angle": round(max_roll or 0.0, 2),
        "avg_confidence": round((avg_conf or 0.0) * 100, 1),
        "stability_rate": stability_rate,
        "warning_frames": warning_count,
        "lost_frames": lost_count,
        "total_anomalies_recorded": total_events,
        "mission_success_score": mission_score,
        "classification": "EXCELLENT" if mission_score >= 85 else ("ACCEPTABLE" if mission_score >= 70 else "CRITICAL_REVIEW_REQUIRED")
    }

@app.get("/mission/export/csv")
def export_blackbox_csv():
    """Kara kutu telemetri verilerini standart CSV formatında doğrudan indirir."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, frame_id, timestamp, roll_angle, pitch_offset, confidence, status FROM telemetry_logs ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["RECORD_ID", "FRAME_ID", "TIMESTAMP_SEC", "ROLL_DEG", "PITCH_OFFSET_PX", "CONFIDENCE", "FLIGHT_STATUS"])
    for row in rows:
        writer.writerow(list(row))

    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tusas_telemetry_blackbox.csv"}
    )

# CANLI KAMERA / HUD VIDEO YAYIN FONKSİYONU
def generate_frames():
    video_path = os.path.join(BASE_DIR, "data", "flight_test.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_id = 0
    start_time = time.time()
    last_event_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            if os.path.exists(video_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_id += 1
        current_time = round(time.time() - start_time, 2)
        frame = cv2.resize(frame, (640, 360))
        
        # 1. Analiz et
        telemetry = processor.process_frame(frame)
        
        # 2. SQLite'a yaz
        log_telemetry(
            frame_id=frame_id,
            timestamp=current_time,
            roll_angle=telemetry["roll"],
            pitch_offset=telemetry["pitch"],
            confidence=telemetry["confidence"],
            status=telemetry["status"]
        )

        # 3. Kritik Durum / Anomali Yakalama ve Fotoğraf Kaydetme (1.5 sn aralıkla)
        if telemetry["status"] in ["WARNING", "LOST"] and (current_time - last_event_time) > 1.5:
            filename = f"event_f{frame_id}_{int(current_time)}.jpg"
            img_disk_path = os.path.join(ANOMALIES_DIR, filename)
            
            # Diske kaydet
            cv2.imwrite(img_disk_path, frame)

            event_type = "HIGH_ROLL_WARNING" if telemetry["status"] == "WARNING" else "HORIZON_LOST"
            # Web üzerinden erişilecek URL yolu
            img_web_path = f"/anomalies/{filename}"
            
            log_event(
                frame_id=frame_id,
                timestamp=current_time,
                event_type=event_type,
                measured_value=telemetry["roll"],
                image_path=img_web_path
            )
            last_event_time = current_time

        # 4. Aviyonik HUD göstergesini kameraya giydir
        hud_frame = hud.draw_hud(frame, telemetry)

        # 5. JPEG formatına çevirip tarayıcıya stream et
        ret, buffer = cv2.imencode('.jpg', hud_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    """Kamera görüntüsünü gerçek zamanlı web stream olarak verir."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")