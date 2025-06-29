import cv2

camera_id = "/dev/video0"  # ganti jika pakai video60
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"❌ Kamera gagal dibuka: {camera_id}")
else:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Kamera terbuka: {camera_id}")
    print(f"  Resolusi: {width} x {height}")
    print(f"  FPS     : {fps}")
    
    # Ambil satu frame untuk uji
    ret, frame = cap.read()
    if ret:
        print(f"  Frame berhasil dibaca. Shape: {frame.shape}")
    else:
        print("❌ Gagal membaca frame.")

cap.release()
