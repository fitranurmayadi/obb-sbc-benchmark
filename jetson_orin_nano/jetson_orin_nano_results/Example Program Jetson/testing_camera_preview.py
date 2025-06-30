import cv2
import time

video_src = "/dev/video0"  # ganti jika perlu

print(f"📷 Mencoba membuka kamera: {video_src}")
cap = cv2.VideoCapture(video_src)

if not cap.isOpened():
    print("❌ Kamera gagal dibuka. Periksa koneksi atau jalur video.")
else:
    print("✅ Kamera berhasil dibuka. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame.")
            break

        cv2.putText(frame, f"Camera: {video_src}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Preview Kamera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️  Preview dihentikan oleh pengguna.")
            break

    cap.release()
    cv2.destroyAllWindows()

