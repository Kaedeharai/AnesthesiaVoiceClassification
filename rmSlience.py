from pydub import AudioSegment
import os


def clean_and_save_audio(input_file_path, output_file_path, threshold=-40.0):
    audio = AudioSegment.from_file(input_file_path, format='wav')
    start = 0
    end = len(audio)

    for i in range(0, len(audio), 10):
        if audio[i:i + 10].dBFS > threshold:
            start = i
            break
    for i in range(len(audio) - 10, -1, -10):
        if audio[i:i + 10].dBFS > threshold:
            end = i + 10
            break 

    cleaned_audio = audio[start:end]
    cleaned_audio.export(output_file_path, format='wav')
    print(f"Successfully processed and saved {input_file_path} to {output_file_path}")


def process_folder(input_folder, output_folder, threshold=-40.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, relative_path)
                output_dir = os.path.dirname(output_file_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                clean_and_save_audio(input_file_path, output_file_path, threshold)


input_folder = "jsmp3"
output_folder = "data/train_jdsu"
process_folder(input_folder, output_folder)