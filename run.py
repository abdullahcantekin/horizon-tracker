import os
import time
import cv2

# Modüllerimizi içe aktarıyoruz
from backend.database import init_db, log_event, log_telemetry
from core.hud import HUDDisplay
from core.processor import HorizonProcessor


def main():
    # 1. Veri tabanını kontrol et ve hazırla
    init_db()

    # 2. Modülleri başlat
    processor = HorizonProcessor()
    hud = HUDDisplay()

    # Video kaynağı: 'data/flight_test.mp4' varsa onu kullanır, yoksa web kamerasını (0) açar
    video_path = os.path.join("data", "flight_test.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"[BILGI] Test videosu oynatılıyor: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print(
            "[BILGI] 'data/flight_test.mp4' bulunamadı, Web Kamerası açılıyor..."
        )

    if not cap.isOpened():
        print("[HATA] Kamera veya video kaynağı açılamadı!")
        return

    frame_id = 0
    start_time = time.time()
    last_event_time = 0

    print("[BILGI] Sistem çalışıyor. Çıkmak için pencerede 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video bittiyse başa sar (döngüsel test için)
            if os.path.exists(video_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_id += 1
        current_time = round(time.time() - start_time, 2)

        # Görüntüyü standart boyuta getir (performans ve tutarlılık için)
        frame = cv2.resize(frame, (800, 480))

        # 3. Görüntü İşleme ve Ufuk Kestirimi
        telemetry = processor.process_frame(frame)

        # 4. Veri Tabanına Telemetri Kaydı
        log_telemetry(
            frame_id=frame_id,
            timestamp=current_time,
            roll_angle=telemetry["roll"],
            pitch_offset=telemetry["pitch"],
            confidence=telemetry["confidence"],
            status=telemetry["status"],
        )

        # 5. Kritik Durum / Anomali Kontrolü (Saniyede en fazla 1 kez tetiklenir)
        if (
            telemetry["status"] in ["WARNING", "LOST"]
            and (current_time - last_event_time) > 1.5
        ):
            anomaly_filename = f"event_f{frame_id}_{int(current_time)}.jpg"
            anomaly_path = os.path.join("exports", "anomalies", anomaly_filename)

            # Anomali görüntüsünü diske kaydet
            cv2.imwrite(anomaly_path, frame)

            event_type = (
                "HIGH_ROLL_WARNING"
                if telemetry["status"] == "WARNING"
                else "HORIZON_LOST"
            )
            log_event(
                frame_id=frame_id,
                timestamp=current_time,
                event_type=event_type,
                measured_value=telemetry["roll"],
                image_path=anomaly_path,
            )
            last_event_time = current_time

        # 6. Aviyonik HUD Göstergelerini Çiz
        display_frame = hud.draw_hud(frame, telemetry)

        # 7. Ekranda Göster
        cv2.imshow("TUSAS - Vision Horizon Tracker (GCS)", display_frame)

        # 'q' tuşuna basıldığında güvenli çıkış yap
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[BILGI] Uçuş simülasyonu sonlandırıldı.")


if __name__ == "__main__":
    main()