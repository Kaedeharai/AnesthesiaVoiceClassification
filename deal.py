from Preprocessing import *
import os
import json
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment


def deal_num(num, folder, name, jsonfile):
    if num == "01":
        label = "before"
        filename = name + "第一次.wav"
        for i in jsonfile:
                if i['data'] == name + "第一次.wav":
                    age = i['age']
                    break
                else:
                    age = "unknown"
    elif num == "02":
        label = "after"
        filename = name + "第二次.wav"
        for i in jsonfile:
                if i["data"] == name + "第二次.wav":
                    age = i["age"]
                    break
                else:
                    age = "unknown"
    elif num == "03":
        label = "recovery"
        filename = name + "第三次.wav"
        for i in jsonfile:
                if i["data"] == name + "第三次.wav":
                    age = i["age"]
                    break
                else:
                    age = "unknown"

    audio = AudioSegment.from_file(os.path.join(folder, filename))
    duration = len(audio) / 1000
    
    return label, age, duration

def batch_process_audio_folder(
    input_folder,
    output_folder,
    feature_type='mfcc',
    sample_rate=22050,
    max_frames=None,
    output_json='features_metadata.json'
):
    os.makedirs(output_folder, exist_ok=True)
    audio_files = []
    for f in os.listdir(input_folder):
        audio_files.append(f)
    progress = tqdm(audio_files, desc=f"Processing {feature_type.upper()} features")
    
    file_dict = []

    for filename in progress:

            filepath = os.path.join(input_folder, filename)
            audio, _ = librosa.load(filepath, sr=sample_rate)

            if feature_type == 'mfcc':
                features, mfccmax, mfccmin = transform_mfcc(audio)
            elif feature_type == 'mel':
                features, melmax, melmin = transform_mel(audio)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}.npy")
            np.save(output_path, features)



            parts = filename.split('_')
            if len(parts) >= 2:
                name = parts[0]
                if parts[1] == 'js':
                    source = "江苏医院"
                    JsonFile = "OriginalData/js.json"
                    Folder = "OriginalData/jdsu"
                    with open(JsonFile, 'r', encoding='utf-8') as f:
                        jsjson = json.load(f)
                    label, age, duration = deal_num(parts[3], Folder, name, jsjson)

                elif parts[1] == 'cg':
                    source = "长庚医院"
                    JsonFile = "OriginalData/cg.json"
                    Folder = "OriginalData/ihgg"
                    with open(JsonFile, 'r', encoding='utf-8') as f:
                        cgjson = json.load(f)
                    for FileName in os.listdir(Folder):
                        label, age, duration = deal_num(parts[3], Folder, name, cgjson)

                elif parts[1] == 'cz':
                    source = "常州医院"
                    JsonFile = "OriginalData/cz.json"
                    Folder = "OriginalData/ihvb"
                    with open(JsonFile, 'r', encoding='utf-8') as f:
                        czjson = json.load(f)
                    for FileName in os.listdir(Folder):
                        label, age, duration = deal_num(parts[3], Folder, name, czjson)

                elif parts[1] == 'xm':
                    source = "厦门医院"
                    JsonFile = "OriginalData/xm.json"
                    Folder = "OriginalData/xwmf"
                    with open(JsonFile, 'r', encoding='utf-8') as f:
                        xmjson = json.load(f)
                    for FileName in os.listdir(Folder):
                        label, age, duration = deal_num(parts[3], Folder, name, xmjson)
                
                if parts[2] == '01':
                    sex = "male"
                elif parts[2] == '02':
                    sex = "female"
                elif parts[2] == '03':
                    sex = "unknown"

                cut = parts[4][:-4]

                file_dict.append({
                'name': name,
                'featrue': filename[:-4] + '.npy',
                'source': source,
                'duration': float(duration),
                'mfcc_maximum': float(mfccmax),
                'mfcc_minimum': float(mfccmin),
                'age': age,
                'sex': sex,
                'label': label,
                'cut': int(cut)
            })
 
        # except Exception as e:
        #     print(f"\nError processing {filename}: {str(e)}")
        #     continue

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(file_dict, f, indent=4, ensure_ascii=False)
        
    print(f"\nProcessing completed. Total processed files: {len(file_dict)}")
    print(f"Features saved to: {output_folder}")
    print(f"Metadata saved to: {output_json}")


batch_process_audio_folder(
    input_folder='train',
    output_folder='mfcc/train',
    feature_type='mfcc',
    sample_rate=22050,
    max_frames=None,
    output_json='mfcc/train.json'
)