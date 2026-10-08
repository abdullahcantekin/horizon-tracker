import cv2
import numpy as np


class HUDDisplay:

    def __init__(self):
        # Renk Paleti (BGR formatında)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)

    def draw_hud(self, frame, telemetry_data):
        """Kamera karesinin üzerine aviyonik HUD göstergelerini çizer."""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Merkez Referans Nişangahı (Kokpit Uçak Sembolü)
        cv2.line(
            annotated_frame, (cx - 40, cy), (cx - 10, cy), self.COLOR_CYAN, 2
        )
        cv2.line(
            annotated_frame, (cx + 10, cy), (cx + 40, cy), self.COLOR_CYAN, 2
        )
        cv2.circle(annotated_frame, (cx, cy), 4, self.COLOR_CYAN, -1)

        # 2. Tespit Edilen Ufuk Çizgisini Çiz
        if telemetry_data["detected"] and telemetry_data["line"] is not None:
            x1, y1, x2, y2 = telemetry_data["line"]
            line_color = (
                self.COLOR_RED
                if telemetry_data["status"] == "WARNING"
                else self.COLOR_GREEN
            )
            cv2.line(annotated_frame, (x1, y1), (x2, y2), line_color, 2)
            cv2.circle(
                annotated_frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, line_color, -1
            )

        # 3. Sol Üst Telemetri Bilgi Paneli (Kutu)
        cv2.rectangle(
            annotated_frame, (10, 10), (280, 130), (20, 20, 20), -1
        )  # Arka plan kutusu
        cv2.rectangle(
            annotated_frame, (10, 10), (280, 130), self.COLOR_CYAN, 1
        )  # Çerçeve

        # Metinleri hazırla
        status_color = (
            self.COLOR_RED
            if telemetry_data["status"] in ["WARNING", "LOST"]
            else self.COLOR_GREEN
        )

        texts = [
            (
                f"STATUS: {telemetry_data['status']}",
                (20, 35),
                status_color,
                0.6,
                2,
            ),
            (
                f"ROLL   : {telemetry_data['roll']:+.1f} deg",
                (20, 60),
                self.COLOR_WHITE,
                0.55,
                1,
            ),
            (
                f"PITCH  : {telemetry_data['pitch']:+.1f} px",
                (20, 85),
                self.COLOR_WHITE,
                0.55,
                1,
            ),
            (
                f"CONF   : %{int(telemetry_data['confidence'] * 100)}",
                (20, 110),
                self.COLOR_CYAN,
                0.55,
                1,
            ),
        ]

        for text, pos, color, scale, thickness in texts:
            cv2.putText(
                annotated_frame,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated_frame