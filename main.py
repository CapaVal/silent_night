import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

def export_file(input_file_name):
    # Deal with m4a file : convert to wav
    wav_filename = ""
    if input_file_name[-4:] == ".m4a":
        AudioSegment.converter = 'C:\\Users\\Valentin\\Desktop\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg'
        track = AudioSegment.from_file(input_file_name, format='m4a')
        wav_filename = input_file_name[:-4] + ".wav"
        file_handle = track.export(wav_filename, format='wav')  # convert .m4a to .wav

    elif input_file_name[-4:] == ".wav":
        wav_filename = input_file_name

    else:
        print("Unrecognized file")
        exit()
    return wav_filename


if __name__ == "__main__":
    g_input_file_name = 'C:\\Users\\Valentin\\Desktop\\Sounds\\Nouvel_enregistrement_28.wav'
    g_wav_file_name = export_file(g_input_file_name)
    g_offset = 0
    g_duration = 360
    raw_wav_bytes, sr = librosa.load(g_wav_file_name, offset=g_offset, duration=g_duration, sr=None)
    raw_wav = list(raw_wav_bytes)
    nb_pts = len(raw_wav)
    time_vector = np.arange(g_offset, g_offset + nb_pts / sr, 1 / sr)

    plt.figure()
    plt.plot(time_vector, raw_wav)
    plt.show()
