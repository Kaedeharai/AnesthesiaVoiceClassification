import os
import numpy as np
from pydub import AudioSegment

def get_duration_wave(file_path):
    audio = AudioSegment.from_file(file_path, "wav")
    return len(audio) / 1000.0

def process_audio(file_path, median):
    audio = AudioSegment.from_file(file_path, "wav")
    audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
    wave_dur = len(audio) / 1000.0
    base_name, ext = os.path.splitext(file_path)

    if wave_dur >= median:
        
        sample_rate = audio.frame_rate
        chunk_samples = int(median * sample_rate)
        overlap_samples = int(chunk_samples * 0.2)

        cut_parameters = np.arange(0, wave_dur, median * 0.8)
        start_time = int(0)
        for i, t in enumerate(cut_parameters):
            if t == 0:
                continue
            stop_time = int(t * 1000 + median * 200)
            if i == len(cut_parameters) - 1:
                stop_time = len(audio)
                start_time = stop_time - median * 1000

            audio_chunk = audio[start_time:stop_time]
            audio_chunk.export(f"{base_name}_{i - 1}{ext}", format="wav")
            start_time = t * 1000

        os.remove(file_path)

    elif wave_dur < median and wave_dur > 0:
        repeat_count = int(np.ceil(median / wave_dur)) + 1
        start_time = int((wave_dur / 2 - wave_dur / 4) * 1000)
        end_time = int((wave_dur / 2 + wave_dur / 4) * 1000)
        middle_chunk = audio[start_time:end_time]
        extended_audio = middle_chunk * 2 * repeat_count
        extended_audio = extended_audio[:median * 1000]
        extended_audio.export(f"{base_name}_0{ext}", format="wav")
        os.remove(file_path)

median = 10

for filename in os.listdir('rmslience/js'):
    file_path = os.path.join('rmslience/js', filename)
    process_audio(file_path, median)
