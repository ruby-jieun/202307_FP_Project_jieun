import numpy as np

# .npy 파일 경로
file_path = "C:/Users/JIEUN/Desktop/202307_FP_Project_jieun-/Data/preprocessing/MOUNTING_labels.npy"

# 파일 읽기
data = np.load(file_path)

# 데이터 확인
print(data)
