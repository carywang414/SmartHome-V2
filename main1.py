import os
import cv2
import face_recognition
import numpy as np
import time

def load_known_faces(whitelist_folder='face'):
    known_encodings = []
    known_names = []

    for filename in os.listdir(whitelist_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(whitelist_folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f"✅ 載入 {name} 的人臉成功")
            else:
                print(f"⚠️ 無法從 {filename} 擷取人臉向量")
    return known_encodings, known_names

def main():
    print("📁 載入白名單人臉...")
    known_encodings, known_names = load_known_faces()

    print("🎥 開啟攝影機...")
    cap = cv2.VideoCapture(0)

    elapsed_time = 0
    last_frame_time = time.time()
    last_seen_name = "No face"
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            # 取第一張臉作為主體
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.3)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    recognized = True

            current_name = name

            # 若切換（含 Unknown 與白名單互換、或 No face → 有臉）
            if current_name != last_seen_name:
                elapsed_time = 0
                print(f"🔁 切換：{last_seen_name} → {current_name}")
                last_seen_name = current_name
            else:
                elapsed_time += now - last_frame_time

            # 畫框
            (top, right, bottom, left) = face_locations[0]
            box_color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            current_name = "No face"
            if current_name != last_seen_name:
                elapsed_time = 0
                print(f"🔁 切換：{last_seen_name} → No face")
                last_seen_name = current_name
            # 不加時間
            cv2.putText(frame, "🚫 無人臉", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        last_frame_time = now

        # 顯示秒數
        title_text = f"detecting {elapsed_time:.1f} s"
        cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Face Recognition", frame)

        if recognized and last_seen_name != "Unknown" and last_seen_name != "No face" and elapsed_time >= 3:
            print("✅ 成功辨識人臉，3秒內通過！")
            break
        if last_seen_name == "Unknown" and elapsed_time >= 10:
            print("❌ 未成功辨識人臉，10秒結束")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 手動結束")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()