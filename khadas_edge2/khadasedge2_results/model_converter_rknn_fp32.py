import os

# Definisi arsitektur dan resolusi
architectures = ["n", "s", "m"]
resolutions = [320, 640, 1280]

# Iterasi kombinasi model
for arch in architectures:
    for res in resolutions:
        model_name = f"yolo11{arch}-obb-{res}.pt"
        command = f"yolo export model={model_name} format=rknn name=rk3588"
        print(f"[INFO] Ekspor model: {model_name} ")
        os.system(command)

