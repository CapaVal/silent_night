import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime
from pydub import AudioSegment
import numpy as np
import scipy
import matplotlib.ticker as ticker
import time

def export_file(input_file_name):
    # Deal with m4a file : convert to wav
    wav_filename = ""
    if input_file_name[-4:] == ".m4a":
        AudioSegment.converter = 'C:\\Users\\Valentin\\Desktop\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg'
        track = AudioSegment.from_file(input_file_name, format='m4a')
        wav_filename = input_file_name[:-4] + ".wav"
        track.export(wav_filename, format='wav')  # convert .m4a to .wav

    elif input_file_name[-4:] == ".wav":
        wav_filename = input_file_name

    else:
        print("Unrecognized file")
        exit()
    return wav_filename

#  This function take a 1-D array, and convert it in 2D array with the first dimension set as nb_pts_sample

def split_raw_wav(raw_wav, nb_pts_sample):
    output_array = []
    nb_pts = len(raw_wav)
    nb_line = (nb_pts // nb_pts_sample)

    for line_index in range(int(nb_line)):
        high_idx_limit = min((line_index + 1) * nb_pts_sample, nb_pts)
        output_array.append(raw_wav[line_index * nb_pts_sample : high_idx_limit])
    return output_array

def get_closest_value_index(freq, sought_value):
    closest_index = 0
    current_diff = 1e9
    for index, value in enumerate(freq):
        if value == sought_value:
            closest_index = index
            break

        if abs(value - sought_value) < current_diff:
            closest_index = index
            current_diff = abs(value - sought_value)

    return closest_index

def process_sound(wav_file_name, offset, duration, sampling_period):
    raw_wav_bytes, sr = librosa.load(wav_file_name, offset=offset, duration=duration, sr=None)
    raw_wav = list(raw_wav_bytes)
    nb_pts = len(raw_wav)
    time_vector = np.arange(offset, offset + nb_pts / sr, 1 / sr)

    sampling_period = 0.2  # The audio time we will use to analyse if there is teeth grinding
    nb_pts_sample = int(sampling_period * sr)
    wav_split_arr = split_raw_wav(raw_wav, nb_pts_sample)
    time_split_arr = split_raw_wav(time_vector, nb_pts_sample)

    window_time = []
    for window_idx in range(len(time_split_arr)):
        window_time.append(time_split_arr[window_idx][0])

    freq_fft = scipy.fftpack.fftfreq(len(wav_split_arr[0]), 1 / sr)
    min_freq = 20
    max_freq = 20000

    min_idx = get_closest_value_index(freq_fft, min_freq)
    max_idx = get_closest_value_index(freq_fft, max_freq)
    freq_fft = freq_fft[min_idx: max_idx + 1]

    res_freq_array = []
    amplitude_array = []
    power_array = []
    for window_idx, windowed_signal in enumerate(wav_split_arr):
        yf = abs(scipy.fft.fft(windowed_signal))
        yf = yf[min_idx: max_idx + 1]
        average_power = 0

        index_max = np.argmax(yf)
        amplitude = yf[index_max]

        for raw_point in windowed_signal[0 : int(len(windowed_signal) / 2)]: # Reduce the average on the first decile of the signal to accelerate processing
            average_power += abs(raw_point)
        average_power /= (len(windowed_signal) / 2)
        average_power *= 1000  # Arbitrary normalization

        if amplitude > 15:
            res_freq = freq_fft[index_max]
        else:
            res_freq = 0

        res_freq_array.append(res_freq)
        amplitude_array.append(amplitude)
        power_array.append(average_power)

        for time_to_display in g_time_to_display:  # Display some FFT and timeseries for debugging pruposes
            if time_split_arr[window_idx][0] <= time_to_display < time_split_arr[window_idx][-1]:
                plt.figure()
                ax = plt.axes()
                ax.plot(time_split_arr[window_idx], windowed_signal)
                ax.set_xlabel("Time in s")
                ax.set_title(f"raw wav for time = {time_to_display} sec")
                ax.set_ylabel("Amplitude")
                ax.xaxis.set_major_formatter(g_fmt_ms)

                plt.figure()
                ax = plt.axes()
                ax.plot(freq_fft, yf)
                ax.set_xlabel("Freq in Hz")
                ax.set_title(f"FFT at time = {time_to_display} sec")
                ax.set_ylabel("FFT Amplitude")

    return window_time, amplitude_array, res_freq_array, power_array

def concatenate_window_data(wav_file_name, window_size, sampling_period):
    total_duration = librosa.get_duration(filename=wav_file_name)
    offset_array = np.arange(0, total_duration - window_size, window_size)

    total_window_time_arr = []
    total_amplitude_arr = []
    total_res_freq_arr = []
    total_power_arr = []

    for offset in offset_array:
        window_time_arr, amplitude_arr, res_freq_arr, power_array = process_sound(wav_file_name, offset, window_size, sampling_period)
        total_window_time_arr +=  window_time_arr
        total_amplitude_arr += amplitude_arr
        total_res_freq_arr += res_freq_arr
        total_power_arr += power_array
    return total_window_time_arr, total_amplitude_arr, total_res_freq_arr, total_power_arr

def smooth_average(data, nb_sample_to_smooth):
    half_nb_sample = int(nb_sample_to_smooth / 2)
    smoothed_value = sum(data[0:half_nb_sample - 1])
    nb_used_sample = half_nb_sample - 1
    smoothed_arr = []

    for idx, new_data in enumerate(data):
        if idx <= half_nb_sample:
            smoothed_value += data[half_nb_sample + idx]
            nb_used_sample += 1
        elif half_nb_sample < idx < len(data) - half_nb_sample:
            smoothed_value += data[half_nb_sample + idx] - data[idx - half_nb_sample - 1]

        elif len(data) - half_nb_sample <= idx:
            smoothed_value -= data[idx - half_nb_sample - 1]
            nb_used_sample -= 1

        smoothed_arr_normalized = smoothed_value / nb_used_sample
        smoothed_arr.append(smoothed_arr_normalized)
    return smoothed_arr

if __name__ == "__main__":
    g_input_file_name = 'C:\\Users\\Valentin\\Desktop\\Sounds\\Nouvel_enregistrement_28.wav'
    g_wav_file_name = export_file(g_input_file_name)
    g_sample_rate = librosa.get_samplerate(g_wav_file_name)

    g_window_size_sec = 30 # Time splitting of the wav file in seconds
    g_time_average_smoothing = 10 # Smoothing of the amplitude to help sound detection
    g_sampling_period = 0.2

    # Set a hh:mm:ss format for the x-axis
    g_fmt = ticker.FuncFormatter(lambda x, pos: time.strftime('%H:%M:%S', time.gmtime(x)))
    # Set a hh:mm:ss:ms format for the x-axis
    g_fmt_ms = ticker.FuncFormatter(lambda x, pos: datetime.utcfromtimestamp(x).strftime('%T.%f')[:-3])

    g_time_to_display = [3600, 5 * 3600 + 34 * 60 + 30, 5 * 3600 + 35 * 60 + 30]

    window_time, amplitude, res_freq_array, g_power_array = concatenate_window_data(g_wav_file_name, g_window_size_sec, g_sampling_period)

    window_size = g_window_size_sec / g_sampling_period # Nb points to apply smoothing
    smoothed_pwr_averaage = smooth_average(g_power_array, window_size)
    plt.figure()
    ax = plt.axes()
    ax.plot(window_time, amplitude)
    ax.set_xlabel("Time in s")
    ax.set_title("Amplitude vs time")
    ax.set_label("Peak ampltitude (a.u.)")
    ax.xaxis.set_major_formatter(g_fmt)

    plt.figure()
    ax = plt.axes()
    ax.plot(window_time, res_freq_array)
    ax.set_xlabel("Time in s")
    ax.set_title("resonnance frequency over time")
    ax.set_ylabel("Fundamental freq (Hz)")
    ax.xaxis.set_major_formatter(g_fmt)

    plt.figure()
    ax = plt.axes()
    ax.plot(window_time, g_power_array)
    ax.set_xlabel("Time in s")
    ax.set_title("power over time")
    ax.set_ylabel("Power (a.u)")
    ax.xaxis.set_major_formatter(g_fmt)

    plt.figure()
    ax = plt.axes()
    ax.plot(window_time, smoothed_pwr_averaage)
    ax.set_xlabel("Time in s")
    ax.set_title(" averaged smoothed power over time")
    ax.set_ylabel("Power (a.u)")
    ax.xaxis.set_major_formatter(g_fmt)

    plt.show()

