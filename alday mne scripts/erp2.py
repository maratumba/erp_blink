import mne
from philistine.mne import abs_threshold, retrieve
import numpy as np

layout = mne.channels.read_layout("EEG1005.lay")
montage = mne.channels.read_montage(kind="standard_1020")

codes = {'long/fit':251, 'long/no-fit':252, 'short/fit':253, 'short/no-fit':254}

# peak-to-peak rejection criteria
peak_to_peak = dict(eeg=150e-6,eog=250e-6)
# flatline rejection criteria
flatline = dict(eeg=5e-6)
# absolute threshold rejection criteria
threshold = 75e-6

# windows of interest for statistics
# units are *milliseconds* relative to time-locking event
wins = dict(baseline=(-200,0),
            N400=(300,500),
            P600=(600,800))

ssavg = {}
for c in codes:
    ssavg[c] = list()

for i in range(5):
    subjfile = 'garfield00{:02d}.vhdr'.format(i+1)
    csvout = 'garfield00{:02d}.csv'.format(i+1)

    raw = mne.io.read_raw_brainvision(subjfile, preload=True)
    # get the channel names and types configured
    mne.rename_channels(raw.info, {'TP9':'A1', 'TP10':'A2', 'PO9':'IO1', 'PO10':'SO1', 'T7':'LO1', 'T8':'LO2'})
    raw.set_eeg_reference(['A1', 'A2'])
    raw.set_channel_types({'IO1':'eog', 'SO1':'eog', 'LO1':'eog', 'LO2':'eog'})

    # zero-phase bandpass FIR filter with passband from 0.1 to 30 Hz
    raw.filter(0.1,30, method='fir', n_jobs=2)

    # extract events
    events = mne.find_events(raw)

    # create epochs
    epochs = mne.Epochs(raw,events=events, event_id=codes,
                        tmin=-0.2, tmax=1.2, # epochs go from -200 to +1200ms
                        detrend=1, # linear detrending
                        baseline=None, # No baselining
                        preload=True)
    epochs.drop_bad()
    bad_epoch_mask = abs_threshold(epochs.pick_types(eeg=True), threshold)
    epochs.drop(bad_epoch_mask,reason="absolute threshold")

    df = retrieve(epochs, wins)
    df.to_csv(csvout)

    for c in codes:
        erp = epochs[c].average()
        ssavg[c].append(erp)
