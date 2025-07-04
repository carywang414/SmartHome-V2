import cv2
import time
from facefunction import load_facedata, process_frame, save_log, save_unknown_image

def main():
    print(" Loading face data...")
    known_encodings, known_names = load_facedata()

    print(" Starting camera...")
    cap = cv2.VideoCapture(0)

    state = {
        "elapsed": 0,
        "last_time": time.time(),
        "last_name": "No face",
        "recognized": False
    }

    last_frame = None
    current_name = "No face"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()  # ← 重要！記得用 .copy()
        frame, current_name = process_frame(frame, known_encodings, known_names, state)
        cv2.imshow("Face Recognition", frame)

        # ✅ 通過：3 秒內辨識白名單
        if state["recognized"] and current_name not in ["Unknown", "No face"] and state["elapsed"] >= 3:
            print(" Face recognized: ", current_name)
            save_log(current_name, "Success")
            break

        #  Unknown timeout：儲存紀錄與截圖
        if current_name == "Unknown" and state["elapsed"] >= 3:
            print(" Unknown face timeout.")
            save_log("Unknown", "Unknown")
            if last_frame is not None:
                save_unknown_image(last_frame)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Manual quit")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()