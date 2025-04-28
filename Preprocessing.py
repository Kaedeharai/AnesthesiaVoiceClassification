import numpy as np
import librosa


def extract(signal, window_size=2048, hop_size=512, mel_bins=128, sample_rate=22050, fmin=0.0, fmax=None, max_frames=None, **kwargs):
    
    stft_matrix = librosa.core.stft(y=signal, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
                               fmax=fmax).T
    
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)
    
    logmel_spc = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spc = logmel_spc.astype(np.float32)
    if max_frames is not None:
        logmel_spc = logmel_spc[0:max(max_frames, len(logmel_spc))]
    
    return logmel_spc[:-1, ]


def mfcc(spectrum, num_filters=48, num_ceps=256):
    
    mfccs = np.zeros(num_ceps)
    mel_spectrum = librosa.feature.melspectrogram(S=spectrum, n_mels=num_filters)
    log_mel_spectrum = librosa.power_to_db(mel_spectrum)
    mfccs = librosa.feature.mfcc(S=log_mel_spectrum, n_mfcc=num_ceps)

    return mfccs


def normalize_features(features):

    min_val = np.min(features)
    max_val = np.max(features)
    normalized_features = (features - min_val) / (max_val - min_val)

    return normalized_features


def transform_mfcc(audio):

    spectrum = np.abs(librosa.stft(audio))
    mfcc_coefficients = mfcc(spectrum)
    normalized_mfcc_coefficients = normalize_features(mfcc_coefficients)

    return normalized_mfcc_coefficients


def transform_mel(audio):

    mel_spectrogram = extract(audio)
    normalized_mel_spectrogram = normalize_features(mel_spectrogram)

    return normalized_mel_spectrogram

