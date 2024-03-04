import math
import librosa
import crepe
import tensorflow as tf
import numpy as np
import pandas as pd
import time

import torch
import harmof0
import torchaudio

import os
import azure.cognitiveservices.speech as speechsdk
import requests
import base64
import json
import time
import numpy as np

import matplotlib.pyplot as plt

import glob

def inference(file_path, reference_text):
    #subscriptionKey = "ec20182c0fe44217ae98437903f03f59"
    subscriptionKey = '0c202f44a3c042e4931af150c72ccd78' #nic
    region = "southeastasia"

    # normal method
    speech_config = speechsdk.SpeechConfig(subscription=subscriptionKey, region=region)
    speech_config.speech_recognition_language = "zh-CN"
    audio_config = speechsdk.AudioConfig(filename=file_path)
    pronunciation_assessment_config = speechsdk.PronunciationAssessmentConfig(
        json_string=f'{{"referenceText":"{reference_text}","gradingSystem":"HundredMark","granularity":"Phoneme","EnableMiscue":"True"}}'
    )

    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    pronunciation_assessment_config.apply_to(speech_recognizer)
    speech_recognition_result = speech_recognizer.recognize_once()
    
    # The pronunciation assessment result as a Speech SDK object
    pronunciation_assessment_result = speechsdk.PronunciationAssessmentResult(
        speech_recognition_result
    )
    print('pronunciation_assessment_result')
    print(pronunciation_assessment_result)
    
    # The pronunciation assessment result as a JSON string
    pronunciation_assessment_result_json = speech_recognition_result.properties.get(
        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
    )
    # Offset and duration is in 100-nanosecond
    # convert it to second, by offset * 1*10^7
    return pronunciation_assessment_result_json


def format_onset_offset(raw_dict):
    onset_times = []
    offset_times = []
    words = []
    for word_dict in json.loads(raw_dict)['NBest'][0]['Words']:
        for phoneme_dict in word_dict['Phonemes']:
            words.append(phoneme_dict['Phoneme'])
            onset_times.append(phoneme_dict['Offset']/10**7)
            offset_times.append((phoneme_dict['Offset']+phoneme_dict['Duration'])/10**7)
    return pd.DataFrame([words, onset_times, offset_times], index = ['word','onset','offset']).transpose()

def get_harmo(audio_path, onset_times = None):
    y, sr = torchaudio.load(audio_path) # harmoF0 requires torchaudio's load
    new_sr = 16000
    y = torchaudio.functional.resample(y, orig_freq = sr, new_freq = new_sr)
    y = torch.mean(y, axis = 0)
    harmo_pitchtracker = harmof0.PitchTracker()

    times = []
    freqs = []
    activations = []
    for character_index in range(len(onset_times)):
        onset_time = onset_times['onset'][character_index]
        offset_time = onset_times['offset'][character_index]
        relevant_frames = y[int(onset_time*new_sr):int(offset_time*new_sr)]
        harmo_time, harmo_freq, harmo_activation, harmo_activation_map = harmo_pitchtracker.pred(relevant_frames, new_sr)
        harmo_time = [t+onset_time for t in harmo_time]
        for t_i, f_i, act_i in zip(harmo_time, harmo_freq, harmo_activation):
            times.append(t_i)
            freqs.append(f_i)
            activations.append(act_i)
    return pd.DataFrame([times, freqs, activations], index=['t', 'f', 'act']).transpose()

def get_crepe(audio_path, onset_times = None, shortest_slice = 10):
    #y, sr = librosa.load(audio_path)
    y_raw, sr = torchaudio.load(audio_path)
    y = torch.mean(y_raw, axis=0).numpy()
    times = []
    f0 = []
    confidence = []
    for character_index in range(len(onset_times)):
        onset_time = onset_times['onset'][character_index]
        offset_time = onset_times['offset'][character_index]
        relevant_frames = y[int(onset_time*sr):int(offset_time*sr)]
        slice_times, slice_f0, slice_confidence, slice_activation = crepe.predict(
            relevant_frames, sr, viterbi=False, #model_capacity="large"
        )
        slice_times = [t+onset_time for t in slice_times]
        for t1, f1, conf1 in zip(slice_times, slice_f0, slice_confidence):
            if f1 > 60:
                times.append(t1)
                f0.append(f1)
                confidence.append(conf1)
    
    # re-scale conf
    print('rescale conf')
    confidence = [x**2 for x in confidence]

    # calculate energy
    print('using energy')
    energy = []
    energy_max = []
    upper_cutoff_freq = 1500
    lower_cutoff_freq = 50
    n_fft = 2048
    hop_length = math.floor(n_fft / 4)
    freq_list = np.arange(0, 1 + n_fft / 2) * sr / n_fft

    to_decibels = torchaudio.transforms.AmplitudeToDB()
    spectrogram = to_decibels(torch.abs(torch.stft(y_raw, n_fft, return_complex=True, hop_length = hop_length))).numpy()
    #spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft = n_fft)), ref=np.max) #512 recommended for speech, may need to change hop length later

    for frame_index in range(len(spectrogram[0])):
        frame_data = [x[frame_index] for x in spectrogram][:sum([a and b for a,b in zip(freq_list < upper_cutoff_freq,freq_list > lower_cutoff_freq)])+1][0]
        energy.append(sum(frame_data))
        energy_max.append(min(max(frame_data), -1))
        #print(frame_index,sum(frame_data)/(np.std(frame_data) * min(max(frame_data), -1)))
    energy_function = [x/(z) for x,z in zip(energy, energy_max)] #if this is above energy_treshold, then we consider it an unvoiced frame
    energy_function = [((x-np.min(energy_function))/(np.max(energy_function)-np.min(energy_function))) for x in energy_function]
    print('energy', np.percentile(energy_function,[0,25,50,75,100]))

    #estimated_times = librosa.frames_to_time(np.arange(len(energy_function)), sr = sr)
    estimated_times = np.arange(len(energy_function)) * hop_length / sr
    energy_table = pd.DataFrame([energy_function, list(estimated_times)], index = ['energy', 'time']).transpose()

    # scale confidence based on energy
    scaled_confidence = []
    for t, conf in zip(times, confidence):
        closest_times = [(t-x)**2 for x in energy_table['time']]
        energy_table['closest'] = closest_times
        energy_table = energy_table.sort_values('closest', ascending = True).reset_index(drop=True)
        mean_energy = np.mean(energy_table[:3]['energy'])
        scaled_confidence.append(conf * mean_energy)
    #renormalise confidence
    cutoff_confidence = 0#0.00001#0.005
    scaled_confidence = [(max((x - min(scaled_confidence))/(max(scaled_confidence - min(scaled_confidence)))-cutoff_confidence, 0))**0.5 for x in scaled_confidence]
    f0_curve = pd.DataFrame([times, f0, scaled_confidence], index = ['t', 'f0', 'conf']).transpose()
    # remove values with zero conf
    f0_curve = f0_curve.loc[f0_curve['conf'] > 0]
    
    # remove outliers with a scaled median for each curve
    print('removing outliers')
    filtered_times = []
    filtered_f0 = []
    filtered_conf = []
    outlier_cutoff = 1.7
    for character_index in range(len(onset_times)): # get distribution for each word
        onset_time = onset_times['onset'][character_index]
        offset_time = onset_times['offset'][character_index]
        word_slice = f0_curve.loc[[a and b for a,b in zip(f0_curve['t'] >= onset_time, f0_curve['t'] <= offset_time)]]
        #calculate weighted median
        median_slice = word_slice.copy(deep=True)
        median_slice.sort_values('f0', ascending = True)
        median_slice['cumulative_conf'] = np.cumsum(median_slice['conf'])
        median_slice = median_slice.loc[median_slice['cumulative_conf'] > 0].reset_index(drop=True)
        median_weight = max(median_slice['cumulative_conf'])/2
        median_slice['distance'] = [(x-median_weight)**2 for x in median_slice['cumulative_conf']]
        median_value = median_slice.loc[median_slice['distance'] == min(median_slice['distance'])]['f0']
        if len(median_value) > 1:
            median_value = np.mean(median_value).item()
        else:
            median_value = median_value.item()
        outlier_slice = word_slice.loc[word_slice['f0'] > outlier_cutoff*median_value]
        inlier_slice = word_slice.loc[word_slice['f0'] <= outlier_cutoff*median_value].reset_index(drop=True)
        for row in range(len(inlier_slice)):
            filtered_times.append(np.round(inlier_slice['t'][row],4))
            filtered_f0.append(inlier_slice['f0'][row])
            filtered_conf.append(inlier_slice['conf'][row])
    
    # do smoothing
    # first, partition into sets of points spaced less than 0.03s apart
    # within each group, each point's f0 is replaced with the average value of all points within +-0.02s
    
    # try the partitioning first
    print('smoothing curve')
    curve_slices = {}
    slice_index = 0

    new_filtered_times = []
    new_filtered_f0 = []
    new_filtered_conf = []
    new_filtered_group = []
    #new_filtered_turns = [] # not useful
    previous_tn = filtered_times[0]
    previous_fn = filtered_f0[0]
    previous_confn = filtered_conf[0]
    temp_times = [previous_tn]
    temp_f0 = [previous_fn]
    temp_conf = [previous_confn]
    temp_group = 0 # indicates which group it is
    for tn, fn, confn in zip(filtered_times[1:], filtered_f0[1:], filtered_conf[1:]):
        d_t = tn - previous_tn
        if d_t > 0.02: # start new 
            if len(temp_times) > shortest_slice:
                # add smoothing algo here 
                smoothing_range = 2
                for i in range(len(temp_times)):
                    lower_bound = max(0,i-smoothing_range)
                    upper_bound = min(len(temp_times)-1,i+smoothing_range)
                    weighted_values = [fn * wn for fn,wn in zip(temp_f0[lower_bound:upper_bound],temp_conf[lower_bound:upper_bound])]
                    #print('conf', temp_conf[lower_bound:upper_bound])
                    #print('values', temp_f0[lower_bound:upper_bound])
                    #print(sum(weighted_values) / sum(temp_conf[lower_bound:upper_bound]))
                    temp_f0[i] = sum(weighted_values) / sum(temp_conf[lower_bound:upper_bound])

                # add to data to send back
                for ti, fi, confi in zip(temp_times, temp_f0, temp_conf):            
                    new_filtered_times.append(ti)
                    new_filtered_f0.append(fi)
                    new_filtered_conf.append(confi)
                    new_filtered_group.append(temp_group)                    
            temp_group += 1
            temp_times = [tn]
            temp_f0 = [fn]
            temp_conf = [confn]
        else:
            temp_times.append(tn)
            temp_f0.append(fn)
            temp_conf.append(confn)
        previous_tn = tn
        previous_fn = fn
        previous_confn = confn

    filtered_f0_curve = pd.DataFrame([new_filtered_times, new_filtered_f0, new_filtered_conf, new_filtered_group], index = ['t', 'f0', 'conf', 'group']).transpose()

    return filtered_f0_curve



def visualise_results(audio_path, methods_dict, cutoff = 2000, filename = None):
    # generate spectrogram data
    y, sr = torchaudio.load(audio_path)
    y = torch.mean(y, axis=0)
    n_fft = 2048
    duration = len(y)/sr
    to_decibels = torchaudio.transforms.AmplitudeToDB(stype = 'power', top_db  = 80)
    spectrogram = to_decibels(torch.abs(torch.stft(y, n_fft, return_complex=True))).numpy()

    # generate y-axis label data
    freq_list = np.arange(0, 1 + n_fft / 2) * sr / n_fft
    cutoff_index = sum([x < cutoff for x in np.arange(0, 1 + n_fft / 2) * sr / n_fft])
    ticks = pd.DataFrame([np.arange(cutoff_index),np.round(freq_list[:cutoff_index])], index=['locs','labels']).transpose()
    tick_scaling = 5
    ticks = ticks.iloc[np.arange(0,cutoff_index, tick_scaling)]

    # generate x-axis label data
    x_ticks = pd.DataFrame([np.arange(len(spectrogram[0])), np.round((duration / len(spectrogram[0]))*np.arange(len(spectrogram[0])),1)], index=['locs','labels']).transpose()
    x_tick_scaling = 100
    x_ticks = x_ticks.iloc[np.arange(0,len(spectrogram[0]), x_tick_scaling)]
      
    # prepare to plot
    fig, ax = plt.subplots(figsize=[10,7])
    # plot background spectrogram
    ax.imshow(spectrogram, origin='lower', aspect='auto', interpolation='nearest', cmap='viridis')
    # plot pitch curves
    for method_name in methods_dict:
        curve_data = methods_dict[method_name]
        # need to map data...
        mapped_freq = curve_data['f'] * n_fft / sr
        mapped_time = (curve_data['t'] / duration) * len(spectrogram[0])
        ax.scatter(mapped_time, mapped_freq, alpha = curve_data['weight'], label = method_name, marker = 'x')
    
    # label the graph
    ax.legend()
    ax.set_ylim([0,cutoff_index])
    ax.set_yticks(ticks['locs'])
    ax.set_yticklabels(ticks['labels'])
    ax.set_xticks(x_ticks['locs'])
    ax.set_xticklabels(x_ticks['labels'])
    plt.show()