import os
import json
import numpy as np
import tensorflow as tf
import urllib.request
import cv2
from PIL import Image
from io import BytesIO


# 텐서플로우에서 사용할 수 있는 형태로 데이터를 변환하는 함수들
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# JSON 파일이 저장된 디렉토리의 경로
json_dir = (
    "C:\\Users\\JIEUN\\Desktop\\202307_FP_Project_jieun\\Data\\Training\\DOG\\MOUNTING"
)

# 이미지와 레이블을 저장할 리스트
images, labels, bounding_boxes = [], [], []

# JSON 파일을 순회하며 데이터를 수집
for root, dirs, files in os.walk(json_dir):
    for file in files:
        if file.endswith(".json"):  # JSON 파일을 찾음
            json_path = os.path.join(root, file)  # JSON 파일의 전체 경로
            with open(json_path, "r", encoding="utf-8") as f:  # JSON 파일 읽기
                data = json.load(f)

            # 각 주석(annotation)에 대한 처리
            for annotation in data["annotations"]:
                emotion = data["metadata"]["inspect"]["emotion"]  # 감정 레이블 추출

                # 객체의 좌표와 크기 정보 추출
                bounding_box = annotation["bounding_box"]
                x = bounding_box["x"]
                y = bounding_box["y"]
                width = bounding_box["width"]
                height = bounding_box["height"]

                # 이미지 URL이 주석에 포함되어 있는 경우
                if "frame_url" in annotation:
                    # 이미지 URL 가져오기
                    img_url = annotation["frame_url"]

                    # URL에서 이미지를 읽기
                    try:
                        resp = urllib.request.urlopen(img_url)
                        img = np.array(Image.open(BytesIO(resp.read())))

                        # 객체 영역만 잘라내기
                        img_crop = img[y : y + height, x : x + width]

                        # 이미지 크기를 재조정
                        img_crop = cv2.resize(img_crop, (224, 224))
                        # 픽셀 값을 [0, 1] 범위로 정규화
                        img_crop = img_crop / 255.0

                        images.append(img_crop)
                        labels.append(emotion)
                        bounding_boxes.append((x, y, width, height))

                    except Exception as e:
                        print(f"Error processing image from URL: {img_url}")
                        print(f"Error message: {str(e)}")

# 리스트를 NumPy 배열로 변환
images = np.array(images)
labels = np.array(labels)
bounding_boxes = np.array(bounding_boxes)

# 전처리된 데이터를 저장할 디렉토리 생성
save_dir = "Data/preprocessing"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 전처리된 이미지, 레이블, 객체의 좌표와 크기를 딕셔너리로 묶기
data = {"images": images, "labels": labels, "bounding_boxes": bounding_boxes}

# 딕셔너리를 NumPy 배열로 저장
np.save(os.path.join(save_dir, "MOUNTING_processed_data.npy"), data)

# 전처리된 이미지를 TFRecord 파일로 저장
tfrecord_file = os.path.join(save_dir, "MOUNTING_processed_images.tfrecords")
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for i in range(len(images)):
        # 이미지를 JPEG 형식으로 인코딩하고, 픽셀 값을 원래 범위로 복원
        img_bytes = tf.io.encode_jpeg((images[i] * 255).astype(np.uint8)).numpy()
        # 레이블을 해시값으로 변환 (충돌 가능성 최소화)
        label = hash(labels[i]) % (10**9)
        # 피쳐 구성
        feature = {
            "image": _bytes_feature(img_bytes),
            "label": _int64_feature(label),
        }
        # TFRecord 메시지 생성
        example_message = tf.train.Example(features=tf.train.Features(feature=feature))
        # 메시지를 TFRecord 파일에 기록
        writer.write(example_message.SerializeToString())

# TFRecord 파일의 크기 출력
# file_size = os.path.getsize(tfrecord_file)
# print(f"TFRecord 파일 크기: {file_size} bytes")
