import os
import cv2
import face_recognition
import numpy as np
import time
import datetime
import pandas as pd

# 載入白名單
def save_log(name, result, excel_path="log.xlsx"):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"Time": now, "Name": name, "Result": result}

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, engine="openpyxl")  # 確保用 openpyxl
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f" Log saved: {row}")

def save_unknown_image(frame, folder="unknown_faces"):
    try:
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"unknown_{timestamp}.jpg")
        success = cv2.imwrite(path, frame)
        if success:
            print(f" Unknown face snapshot saved to: {path}")
        else:
            print(f" Failed to save image to: {path}")
    except Exception as e:
        print(f" Error saving unknown image: {e}")
def load_known_faces(folder='face'):
    known_encodings = []
    known_names = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f" 載入 {name} 的人臉成功")
            else:
                print(f" 無法從 {filename} 擷取人臉向量")
    return known_encodings, known_names

# 處理一幀影像，回傳辨識狀態與更新秒數
def process_frame(frame, known_encodings, known_names, state):
    now = time.time()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_name = "No face"
    name = "No face"

    if face_encodings:
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                state["recognized"] = True
        current_name = name

        # 比對人名變化以重置秒數
        if current_name != state["last_name"]:
            state["elapsed"] = 0
            print(f" 切換：{state['last_name']} → {current_name}")
            state["last_name"] = current_name
        else:
            state["elapsed"] += now - state["last_time"]

        # 畫框
        (top, right, bottom, left) = face_locations[0]
        box_color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    else:
        current_name = "No face"
        if current_name != state["last_name"]:
            state["elapsed"] = 0
            print(f" 切換：{state['last_name']} → No face")
            state["last_name"] = current_name
        # 不累加秒數
        cv2.putText(frame, " 無人臉", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 畫標題狀態列
    elapsed = state["elapsed"]
    if current_name == "No face":
        title_text = " No Face"
        title_color = (200, 200, 200)
    elif current_name == "Unknown":
        title_text = f"Unknown... {elapsed:.1f} s"
        title_color = (0, 0, 255)
    else:
        title_text = f" passing ... {elapsed:.1f} s"
        title_color = (0, 255, 0)

    cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, title_color, 2, cv2.LINE_AA)

    state["last_time"] = now
    return frame, current_name

def load_facedata(folder="facedata"):
    known_encodings = []
    known_names = []

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            vector = np.load(os.path.join(folder, file))
            known_encodings.append(vector)
            known_names.append(name)
            print(f" 載入 {name} 向量成功")
    return known_encodings, known_names