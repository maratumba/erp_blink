#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Plot events and blinks.')
parser.add_argument('vhdr_file', help='vhdr file name')
args = parser.parse_args()

import os.path as op
import numpy as np
import mne

try:
    raw=mne.io.read_raw_brainvision(args.vhdr_file, preload=True)
except:
    raise OSError('Failed to read file: {}'.format(args.vhdr_file))

mne.add_reference_channels(raw, 'LiRef', copy=False)
raw.set_eeg_reference(['ReRef','LiRef'])
raw.filter(0.1,30, method='fir', picks=[27,28,29,30]) 

event_id = {'sem-yes-x': 203,
    'sem-no-x': 208,
    'world-yes-x': 213,
    'world-no-x': 218,
    'rel-yes-x': 223,
    'rel-no-x': 228,
    'abs—min-yes-x': 233,
    'abs—min-no-x': 238,
    'abs—max-yes-x': 243,
    'abs—max-no-x': 248}

raw.set_channel_types({'EOGli':'eog','EOGre':'eog','EOGobre':'eog','EOGunre':'eog'})
picks = mne.pick_types(raw.info,eog=True)
events = mne.find_events(raw)

rev_event_dict = {
    'sem-yes': 200, 'sem-yes-das': 201, 'sem-yes-ist': 202,'sem-yes-x': 203,
    'sem-no': 205, 'sem-no-das': 206, 'sem-no-ist': 207,'sem-no-x': 208,
    'world-yes': 210, 'world-yes-das': 211, 'world-yes-ist': 212,'world-yes-x': 213,
    'world-no': 215, 'world-no-das': 216, 'world-no-ist': 217,'world-no-x': 218,
    'rel-yes': 220, 'rel-yes-das': 221, 'rel-yes-ist': 222,'rel-yes-x': 223,
    'rel-no': 225, 'rel-no-das': 226, 'rel-no-ist': 227,'rel-no-x': 228,
    'abs—min-yes': 230, 'abs—min-yes-das': 231, 'abs—min-yes-ist': 232,'abs—min-yes-x': 233,
    'abs—min-no': 235, 'abs—min-no-das': 236, 'abs—min-no-ist': 237,'abs—min-no-x': 238,
    'abs—max-yes': 240, 'abs—max-yes-das': 241, 'abs—max-yes-ist': 242,'abs—max-yes-x': 243,
    'abs—max-no': 245, 'abs—max-no-das': 246, 'abs—max-no-ist': 247,'abs—max-no-x': 248,
    'display': 195, 'correct': 196, 'incorrect': 197, 'timeout': 199, 
    'block1': 181, 'block2': 182, 'block3': 183, 'block4': 184, 'block5': 185, 'block6': 186, 'block7': 187
}
event_dict ={v: k for k, v in rev_event_dict.items()}

# Construct event annotations:
event_annotations=[]
for event in events:
    if event[2] in event_dict.keys():
        event_annotations.append(event_dict[event[2]])
    else:
        event_annotations.append(str(event[2]))


eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
n_events = len(events)
# Center to cover the whole blink with full duration of 0.5s:
onset_bl = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration_bl = np.repeat(0.5, n_blinks)

onset_ev = events[:, 0] / raw.info['sfreq'] - 0.25
duration_ev = np.repeat(0.5, n_events)

onset = np.hstack((onset_bl,onset_ev))
duration = np.hstack((duration_bl,duration_ev))

comb_events = np.vstack((eog_events,events))
raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks + event_annotations,
                                  orig_time=raw.info['meas_date'])
print(raw.annotations)  # to get information about what annotations we have
raw.plot(events=comb_events)  # To see the annotated segments.