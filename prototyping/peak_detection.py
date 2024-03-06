import tensorflow
import crepe

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import interpolate
import time
from matplotlib import cm

class real_time_peak_detection():
    def __init__(self, array, lag, threshold, influence):
        """ 
        these are the params for detection
        lag is the time interval that it 'scans' over
        threshold is how many standard deviations away a point has to be before it's considered a 'peak'
        influence is a scaling factor to reduce how much the 'peak' values affect the moving average. If you set it to 0 then the moving average will be static during time ranges where a peak is detected.

        signals is the input data. I honestly think it's kinda inefficient how this is set up, but I think it's so that it can handle real-time data. Didn't feel like making a new method just for quick testing so I left it as is. The function will only tekll you whether the next point is a peak or not.
        """
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag 
        self.threshold = threshold
        self.influence = influence
        """
        so the next few lines here is to set up the initial values (because it can't start finding a 'moving average' until it has enough values to average over)
        """
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        """
        this is to handle the next input
        """
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag: # as mentioned earlier - can't start finding a 'moving average' until there's enough values
            return 0
        elif i == self.lag: # when there's enough values then it starts to find moving average
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag] = np.std(self.y[0:self.lag]).tolist()
            return 0

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > (self.threshold * self.stdFilter[i - 1]): # then check if the value exceeds the threshold for peak detection, if it does, then it registers it as a 'peak range' and returns back 

            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + \
                (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i], self.avgFilter[i], self.stdFilter[i]

def get_peak_ranges(frame_data, window = 10, threshold = 0.7, influence = 0.5, freq_list = None, show=False, title = None):
    """
    so the point of this function is to scan the frame data (i.e. the amplitudes of all frequencies for a specific single time frame) to find the time ranges where it is a peak.
    why I need this is because the function I grabbed from stackoverflow returns the _ranges_ for which there is a peak, doesn't return the actual point of the peak.
    """
    # input is the spectrum for one time-frame
    # output is the set of coordinates
    peak = []
    moving_averages = []
    st_devs = []
    for i in range(int(window/2)):
        peak.append(0)
        moving_averages.append(frame_data[i])
        st_devs.append(0)

    for i in range(len(frame_data)-(window)):
        current_window = frame_data[i:i+window]
        
        if window %2 == 0: #even
            next_value = (current_window[int(window/2)] + current_window[int((window/2)-1)])/2
        else: #odd
            next_value = current_window[int((window-1)/2)] 
        # I honestly don't remember why I did this odd/even thing...

        moving_average_calculator = real_time_peak_detection(current_window, lag = window, threshold = threshold, influence = influence) # this tells me whether the next point is a peak
        moving_average_calculator.thresholding_algo(next_value)
        peak_detect, moving_average, st_dev = moving_average_calculator.thresholding_algo(next_value)
        # the next 3 lines are just to set up 3 lists to correspond to: is this timeframe a peak or not, what is the moving average for the timeframe, what is the stdev for the timeframe
        # actually I don't need to collect the list of moving average and stdev. The real_time_peak_detection method already makes use of it internally. But I wanted this data to plot the graph later.
        peak.append(peak_detect)
        moving_averages.append(moving_average)
        st_devs.append(st_dev*threshold)

    if show:
        fig, ax = plt.subplots(figsize=[15,5])
        ax.plot(freq_list[:len(moving_averages)], moving_averages, color = 'red')
        ax.plot(freq_list[:len(moving_averages)], [x+y for x,y in zip(moving_averages,st_devs)], color = 'red', alpha=0.5) # this is the upper range (above this range is a 'peak')
        ax.plot(freq_list[:len(moving_averages)], [x-y for x,y in zip(moving_averages,st_devs)], color = 'red', alpha=0.5) # lower range. This would be considered a trough (anti-peak) but in this case we don't care.

        ax.plot(freq_list, frame_data, color='black')
        
        if title != None:
            ax.set_title(title)

        #ticks = np.arange(4,15)
        #ax.set_xticks(ticks)
        #ax.set_xticklabels(2**ticks)
        ax.set_xlim([0,2000])
        plt.show()

    return peak, moving_averages, st_devs

# define the function to get peaks
# we need this because the earlier function doesn't say exactly where the peak is
# so I iterate over the frequencies one by one to find peak ranges, then in each range, I grab the highest point and say that's the coordinates of the peak.

def find_peaks(peak, frame_data, freq_list):
    peak_coordinates = []
    peaks = []
    freqs = []
    for i in range(len(peak)):
        if peak[i] != 1:
            if len(peaks) != 0:
                peak_coordinates.append([max(peaks),freqs[peaks.index(max(peaks))]])
            peaks = []
            freqs = []
        elif peak[i] == 1:
            peaks.append(frame_data[i])
            freqs.append(freq_list[i])
    if len(peak_coordinates) == 0:
        return None
    else:
        return peak_coordinates


def find_raw_peak(audio_path, onset_times, cutoff_freq):
    y, sr = librosa.load(audio_path)
    
    #using default n_fft
    n_fft = 2048
    freq_list = np.arange(0, 1 + n_fft / 2) * sr / n_fft

    #get spectrogram
    spectrogram_start = time.time()
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft = n_fft)), ref=np.max) #512 recommended for speech, may need to change hop length later

    #get peaks
    estimated_f0s = []
    estimated_frames = []
    losses = []
    allowed_freq_ranges = []
    test_peaks_cleaned_list = []
    display_output_loss_list = []
    all_peaks = []

    for frame_index in range(len(spectrogram[0])):
        frame_data = [x[frame_index] for x in spectrogram]
        peak, moving_averages, st_devs = get_peak_ranges(frame_data, show = False, freq_list = freq_list, title = str(frame_index)+": "+str(librosa.frames_to_time(frame_index, sr = sr)))
        peak_coordinates = find_peaks(peak, frame_data, freq_list)
        all_peaks.append(peak)
        estimated_frames.append(frame_index)
        test_peaks_cleaned_list.append(peak_coordinates)

    estimated_times = librosa.frames_to_time(estimated_frames, sr = sr) #may need to change hop length

    
    peak_coordinates = []
    peak_times = []
    peak_heights = []
    for cleaned_peaks, time_frame in zip(test_peaks_cleaned_list, estimated_times):
        if cleaned_peaks!= None:
            for row_index in range(len(onset_times)):
                if (time_frame >= onset_times['onset'][row_index]) and (time_frame <= onset_times['offset'][row_index]):
                    for peak_value in cleaned_peaks:
                        if peak_value[1] < 3000:
                            peak_coordinates.append(peak_value[1])
                            peak_heights.append(min(1,(100+peak_value[0])/100))
                            peak_times.append(time_frame)
                    break
            
    return pd.DataFrame([peak_times, peak_coordinates, peak_heights], index=['t','f','h']).transpose()