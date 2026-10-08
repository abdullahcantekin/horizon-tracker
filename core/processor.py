import cv2
import numpy as np


class HorizonProcessor:

    def __init__(self):
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.alpha = 0.75

    def _verify_sky_ground_contrast(self, hsv_frame, x1, y1, x2, y2, h, w):
        """Çizginin üst bölgesinin (gökyüzü) alt bölgesine (zemin) kıyasla

        parlaklık (V kanalı) kontrastını analiz eder.
        """
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)

        offset = 20
        sky_y = max(mid_y - offset, 0)
        ground_y = min(mid_y + offset, h - 1)
        sample_x = max(min(mid_x, w - 1), 0)

        # V (Value/Parlaklık) kanallarını karşılaştır
        sky_brightness = float(hsv_frame[sky_y, sample_x][2])
        ground_brightness = float(hsv_frame[ground_y, sample_x][2])

        # Gökyüzü genel olarak zeminden daha parlaktır
        contrast_diff = sky_brightness - ground_brightness
        return contrast_diff

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        center_y = h // 2

        # 1. Griye çevirme, Gauss Filtresi ve Renk Doğrulaması için HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Canny Kenar Algılama
        edges = cv2.Canny(blurred, 50, 150)

        # 3. Olasılıksal Hough Çizgi Dönüşümü
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(w * 0.20),
            maxLineGap=20,
        )

        detected_line = None
        best_score = -9999.0
        selected_length = 0.0

        if lines is not None:
            for l in lines:
                # Boyut uyumsuzluğunu önlemek için güvenli düzleştirme
                coords = l.ravel()
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = (
                    int(coords[0]),
                    int(coords[1]),
                    int(coords[2]),
                    int(coords[3]),
                )

                length = np.hypot(x2 - x1, y2 - y1)

                # Dikey çizgileri ele (ufuk çizgisi yatay eksendedir)
                angle_deg = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

                if abs(angle_deg) < 70:
                    # HSV Gökyüzü-Yer Renk ve Parlaklık Doğrulaması
                    contrast_diff = self._verify_sky_ground_contrast(
                        hsv, x1, y1, x2, y2, h, w
                    )

                    # Çizgi uzunluğu ile renk ayrımını birleştiren doğrulama skoru
                    candidate_score = length + (contrast_diff * 0.8)

                    if candidate_score > best_score:
                        best_score = candidate_score
                        selected_length = length
                        detected_line = (x1, y1, x2, y2)

        # 4. Açı ve Sapma Hesabı
        if detected_line is not None:
            x1, y1, x2, y2 = detected_line
            raw_roll = float(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            line_mid_y = (y1 + y2) / 2.0
            raw_pitch = float(center_y - line_mid_y)

            # Titreme Önleyici Filtre
            roll_angle = (self.alpha * raw_roll) + (
                (1.0 - self.alpha) * self.prev_roll
            )
            pitch_offset = (self.alpha * raw_pitch) + (
                (1.0 - self.alpha) * self.prev_pitch
            )

            self.prev_roll = roll_angle
            self.prev_pitch = pitch_offset

            confidence = min(round(float(selected_length / w), 2), 1.0)
            status = "WARNING" if abs(roll_angle) > 25 else "STABLE"

            return {
                "detected": True,
                "roll": round(roll_angle, 2),
                "pitch": round(pitch_offset, 2),
                "confidence": confidence,
                "status": status,
                "line": detected_line,
            }

        return {
            "detected": False,
            "roll": self.prev_roll,
            "pitch": self.prev_pitch,
            "confidence": 0.0,
            "status": "LOST",
            "line": None,
        }