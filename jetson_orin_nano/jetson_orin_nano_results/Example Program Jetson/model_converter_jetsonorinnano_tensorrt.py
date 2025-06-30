import os

# Arsitektur model dan resolusi input
#architectures = ["n", "s", "m"]
#resolutions = [320, 640, 1280]


architectures = ["s"]
resolutions = [320]


data_yaml = "../../datasets/obb_mini_things-3/data.yaml"

# Format yang akan diekspor (prioritaskan ncnn)
formats = ["engine"]

# Iterasi semua kombinasi model dan format
for fmt in formats:
    for arch in architectures:
        for res in resolutions:
            model_name = f"yolo11{arch}-obb-{res}.pt"
            if fmt == "engine":
                save_name = f"yolo11{arch}-obb-{res}"
                command = (
                    f"yolo export model={model_name} data={data_yaml} "
                    f"format=engine device=0 imgsz={res} half=True name={save_name}"
                )
            elif fmt == "ncnn":
                save_name = f"yolo11{arch}-obb-{res}_ncnn_model"
                command = (
                    f"yolo export model={model_name} data={data_yaml} "
                    f"format=ncnn half=True name={save_name}"
                )
            else:
                continue

            print(f"[INFO] Exporting {model_name} → {fmt.upper()}")
            os.system(command)

