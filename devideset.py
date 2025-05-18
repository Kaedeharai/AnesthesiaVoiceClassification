import os
import shutil
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


data_dir = "rmslience/js"
SEED = 42

# 收集所有音频文件路径
file_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            file_paths.append(os.path.join(root, file))

# 按说话人分组
speaker_files = defaultdict(list)
for path in file_paths:
    filename = os.path.basename(path)
    speaker = filename.split("_")[0]
    speaker_files[speaker].append(path)

# 划分说话人
speakers = list(speaker_files.keys())
train_speakers, temp_speakers = train_test_split(speakers, test_size=0.35, random_state=SEED)
val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=SEED)


# 生成最终路径列表
train_files = []
val_files = []
test_files = []
for speaker in train_speakers:
    train_files.extend(speaker_files[speaker])
for speaker in val_speakers:
    val_files.extend(speaker_files[speaker])
for speaker in test_speakers:
    test_files.extend(speaker_files[speaker])

print("训练集样本数:", len(train_files))
print("验证集样本数:", len(val_files))
print("测试集样本数:", len(test_files))


def copy_files(file_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for src_path in file_list:
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy(src_path, dst_path)

copy_files(train_files, "AnesthesiaVoiceClassification/train")
copy_files(val_files, "AnesthesiaVoiceClassification/val")
copy_files(test_files, "AnesthesiaVoiceClassification/test")