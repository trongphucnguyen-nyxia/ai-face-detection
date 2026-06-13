import cv2
import mediapipe as mp
import os
import time

input_folder = "test_images"
output_folder = "results"

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_image(path):
    image = cv2.imread(path)

    if image is None:
        print("Error reading:", path)
        return

    # 🔥 resize image (important fix)
    image = cv2.resize(image, (600, 600))

    start = time.time()

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    ) as detector:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        count = 0

        if results.detections:
            for detection in results.detections:
                count += 1
                mp_drawing.draw_detection(image, detection)

        runtime = time.time() - start

        text = f"Faces: {count} | Time: {runtime:.3f}s"

        # clean header bar
        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        output_path = os.path.join(output_folder, "result_" + os.path.basename(path))
        cv2.imwrite(output_path, image)

        print(f"{path} -> Faces: {count}, Time: {runtime:.3f}s")

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(input_folder, file))
