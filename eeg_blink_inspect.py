import os.path as op
import numpy as np

import mne

# Load the file and add a virtual channel for re-referencing:

raw=mne.io.read_raw_brainvision('Obj0002.vhdr',preload=True)
mne.add_reference_channels(raw, 'LiRef', copy=False)
raw.set_eeg_reference(['ReRef','LiRef'])

raw.filter(0.1,30, method='fir') 

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

tmin, tmax = -0.2, 0.5
baseline = (None, 0.0)
raw.set_channel_types({'EOGli':'eog','EOGre':'eog','EOGobre':'eog','EOGunre':'eog'})
reject = {'eog': 40e-6}

picks = mne.pick_types(raw.info,eog=True)
events = mne.find_events(raw)

###############################################################################
######################## Plot with annotated blinks ###########################
eog_events = mne.preprocessing.find_eog_events(raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                  orig_time=raw.info['meas_date'])
print(raw.annotations)  # to get information about what annotations we have
raw.plot(events=eog_events)  # To see the annotated segments.
###############################################################################


# Calculate epochs
epochs = mne.Epochs(raw, 
                    events=events, 
                    event_id=event_id, 
                    tmin=tmin,
                    tmax=tmax, 
                    baseline=None, 
                    detrend=0,
                    reject=reject, 
                    picks=picks)

epochs.drop_bad()
epochs.plot_drop_log()

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