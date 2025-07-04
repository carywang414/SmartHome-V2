import cv2
from facefunction import load_facedata, process_frame
import time

def main():
    print("📂 載入 facedata 向量...")
    known_encodings, known_names = load_facedata()

    print("🎥 開啟攝影機...")
    cap = cv2.VideoCapture(0)

    state = {
        "elapsed": 0,
        "last_time": time.time(),
        "last_name": "No face",
        "recognized": False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, current_name = process_frame(frame, known_encodings, known_names, state)
        cv2.imshow("Face Recognition", frame)

        if state["recognized"] and current_name not in ["Unknown", "No face"] and state["elapsed"] >= 3:
            print("✅ 成功辨識人臉，3秒內通過！")
            break
        if current_name == "Unknown" and state["elapsed"] >= 10:
            print("❌ 未成功辨識人臉，10秒結束")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 手動結束")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()