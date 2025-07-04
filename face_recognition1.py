import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import os
from sklearn.preprocessing import Normalizer
from keras_facenet import FaceNet

# 載入模型
embedder = FaceNet()
model = embedder.model
l2_normalizer = Normalizer('l2')
mp_face_detection = mp.solutions.face_detection

def extract_face(image, center_only=True, draw_box=True):
    """ 偵測人臉並裁切，回傳臉部與畫好框的畫面 """
    image_copy = image.copy()  # 保留畫面給框線用
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        # 對比增強處理
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
        center_rect = [w//3, h//3, 2*w//3, 2*h//3] if center_only else [0, 0, w, h]

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = x1 + w_box, y1 + h_box

                cx, cy = x1 + w_box//2, y1 + h_box//2
                if center_only and not (center_rect[0] < cx < center_rect[2] and center_rect[1] < cy < center_rect[3]):
                    continue

                # ➤ 繪製框線
                if draw_box:
                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

                face_crop = cv2.resize(image[y1:y2, x1:x2], (160, 160))
                return face_crop, image_copy
        enhanced = adjust_gamma(enhanced, gamma=1.5)

    return None, image

def get_embedding(face_img):
    """ 把臉部圖像轉為128維向量 """
    face = face_img.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)[0]
    return l2_normalizer.transform([embedding])[0]

def build_white_list_embeddings(white_folder='white'):
    """將同一人多張圖片的 embeddings 合併為平均向量"""
    database = {}
    grouped = {}  # key: 人名, value: list of embeddings

    for filename in os.listdir(white_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(white_folder, filename)
            image = cv2.imread(path)
            face, image_with_box = extract_face(image)
            if face is not None:
                embedding = get_embedding(face)
                # 支援同名多張，例如 myface1_1.jpg、myface1_2.jpg → name = myface1
                name = os.path.splitext(filename)[0].split('_')[0]
                if name not in grouped:
                    grouped[name] = []
                grouped[name].append(embedding)
            else:
                print(f"[警告] 無法從 {filename} 擷取人臉")

    for name, embeddings in grouped.items():
        avg_embedding = np.mean(embeddings, axis=0)  # 可改 median 更穩定
        database[name] = avg_embedding

    return database

def recognize_face(image, database, threshold=0.7):
    """ 與 white list 比對，返回最接近的名字與距離 """
    face, image_with_box = extract_face(image)
    if face is None:
        return "No face", None,image_with_box
    embedding = get_embedding(face)
    min_dist = float('inf')
    identity = "Unknown"
    for name, db_emb in database.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > threshold:
        return "Unknown", min_dist, image_with_box
    return identity, min_dist, image_with_box

def adjust_gamma(image, gamma=1.5):#亮度處理
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)