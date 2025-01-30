import os
import subprocess
import chardet

def convert_m4a_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index in os.listdir(input_folder):
        for file in os.listdir(os.path.join(input_folder, index)):
            # 如果你想转换其他格式的音频文件，可以修改.m4a为其他格式
            if file.endswith(".m4a") and not file.startswith("._"):
                input_path = os.path.join(os.path.join(input_folder, index), file)
                output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".wav")
                print(f"Converting {input_path} to {output_path}")

                with open(input_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']

                result = subprocess.run(['ffmpeg', '-i', input_path, output_path], capture_output=True, text=True, encoding=encoding)
                if result.returncode != 0:
                    print(f"Error converting {input_path}: {result.stderr}")
                else:
                    print(f"Successfully converted {input_path} to {output_path}")

input_folder = 'ttt'
output_folder = 'jdsu'
convert_m4a_to_wav(input_folder, output_folder)