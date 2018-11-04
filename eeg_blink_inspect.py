import os.path as op
import numpy as np

import mne

# Load the file and add a virtual channel for re-referencing:

raw=mne.io.read_raw_brainvision('Obj0002.vhdr',preload=True)
mne.add_reference_channels(raw, 'LiRef', copy=False)
raw.set_eeg_reference(['ReRef','LiRef'])

raw.filter(0.1,30, method='fir') 

event_adjectives = {'sem-yes-x': 203,
    'sem-no-x': 208,
    'world-yes-x': 213,
    'world-no-x': 218,
    'rel-yes-x': 223,
    'rel-no-x': 228,
    'abs—min-yes-x': 233,
    'abs—min-no-x': 238,
    'abs—max-yes-x': 243,
    'abs—max-no-x': 248,
    'filler-no':249
    }

tmin, tmax = -0.2, 0.5
baseline = (None, 0.0)
raw.set_channel_types({'EOGli':'eog','EOGre':'eog','EOGobre':'eog','EOGunre':'eog'})
reject = {'eog': 40e-6}

eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=raw.info['meas_date'])
print(raw.annotations)  # to get information about what annotations we have
raw.plot(events=eog_events)  # To see the annotated segments.

picks = mne.pick_types(raw.info,eeg=True,eog=True)
events = mne.find_events(raw)

#######################
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


# Calculate epochs
epochs = mne.Epochs(raw, 
                    events=events, 
                    event_id=event_adjectives, 
                    tmin=tmin,
                    tmax=tmax, 
                    baseline=None, 
                    detrend=0,
                    picks=picks,
                    #reject=reject,
                    reject_by_annotation=True)

#epochs.drop_bad()
#epochs.plot_drop_log()

epochs.plot()


'''
thresholds = np.arange(10,200,10) * 1e-6
drop_rate1 = []
for threshold in thresholds:
    reject = {'eog': threshold}
    #reject = {'eog': 200e-6}
    picks = mne.pick_types(raw.info,eog=True)
    events = mne.find_events(raw)
    # Calculate epochs
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=baseline, reject=reject, picks=picks)
    epochs.drop_bad()
    epochs.plot_drop_log()
    drop_rate1.append(epochs.drop_log_stats())

import matplotlib.pyplot as plt
plt.plot(thresholds,drop_rate)
plt.show()
'''