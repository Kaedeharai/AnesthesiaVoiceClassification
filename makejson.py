import os
import json
import numpy as np
import librosa
from pydub import AudioSegment


def create_filenames_with_labels_json(data_dir, json_file):
    file_dict = []
 
    for filename in os.listdir(data_dir):
        parts = filename.split('_')
        if len(parts) >= 2:
            name = parts[0]

            if parts[1] == 'js':
                source = "江苏医院"
                JsonFile = "OriginalData/js.json"
                with open(JsonFile, 'r', encoding='utf-8') as f:
                    jsjson = json.load(f)

                for FileName in os.listdir("OriginalData/jdsu"):
                    if parts[3] == "01":
                        label = "before"
                        if FileName == name + "第一次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/jdsu", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in jsjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
                            
                    elif parts[3] == "02":
                        label = "after"
                        if FileName == name + "第二次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/jdsu", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in jsjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
                    elif parts[3] == "03":
                        label = "recovery"
                        if FileName == name + "第三次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/jdsu", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in jsjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
            elif parts[1] == 'cg':
                source = "长庚医院"
                JsonFile = "OriginalData/cg.json"
                with open(JsonFile, 'r', encoding='utf-8') as f:
                    cgjson = json.load(f)

                for FileName in os.listdir("OriginalData/ihgg"):
                    if parts[3] == "01":
                        label = "before"
                        if FileName == name + "第一次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihgg", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in cgjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
                            
                    elif parts[3] == "02":
                        label = "after"
                        if FileName == name + "第二次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihgg", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in cgjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
                    elif parts[3] == "03":
                        label = "recovery"
                        if FileName == name + "第三次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihgg", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in cgjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
            elif parts[1] == 'cz':
                source = "常州医院"
                JsonFile = "OriginalData/cz.json"
                with open(JsonFile, 'r', encoding='utf-8') as f:
                    czjson = json.load(f)

                for FileName in os.listdir("OriginalData/ihvb"):
                    if parts[3] == "01":
                        label = "before"
                        if FileName == name + "第一次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihvb", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in czjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
                             
                    elif parts[3] == "02":
                        label = "after"
                        if FileName == name + "第二次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihvb", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in czjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
                    elif parts[3] == "03":
                        label = "recovery"
                        if FileName == name + "第三次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/ihvb", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in czjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
            elif parts[1] == 'xm':
                source = "厦门医院"
                JsonFile = "OriginalData/xm.json"
                with open(JsonFile, 'r', encoding='utf-8') as f:
                    xmjson = json.load(f)

                for FileName in os.listdir("OriginalData/xwmf"):
                    if parts[3] == "01":
                        label = "before"
                        if FileName == name + "第一次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/xwmf", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in xmjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"

                    elif parts[3] == "02":
                        label = "after"
                        if FileName == name + "第二次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/xwmf", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in xmjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   
                    elif parts[3] == "03":
                        label = "recovery"
                        if FileName == name + "第三次.wav":
                            audio = AudioSegment.from_file(os.path.join("OriginalData/xwmf", FileName))
                            duration = len(audio) / 1000
                            samples = np.array(audio.get_array_of_samples())
                            maximum = np.max(samples)
                            minimum = np.min(samples)
                            for i in xmjson:
                                if i["data"] == FileName:
                                    age = i["age"]
                                    break
                                else:
                                    age = "未知"
   

            if parts[2] == '01':
                sex = "男"
            elif parts[2] == '02':
                sex = "女"
            elif parts[2] == '03':
                sex = "未知"

            cut = parts[4][:1]

            file_dict.append({
                'name': name,
                'featrue': filename,
                'source': source,
                'duration': float(duration),
                'maximum': float(maximum),
                'minimum': float(minimum),
                'age': age,
                'sex': sex,
                'label': label,
                'cut': int(cut)
            })
 
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(file_dict, f, indent=4, ensure_ascii=False)


data_directory = 'mfcc/train'
json_output_file = 'mfcc/train.json'
create_filenames_with_labels_json(data_directory, json_output_file)

