import mne
from philistine.mne import abs_threshold, retrieve
import numpy as np

layout = mne.channels.read_layout("EEG1005.lay")
montage = mne.channels.read_montage(kind="standard_1020")

codes = {'long/fit':251, 'long/no-fit':252, 'short/fit':253, 'short/no-fit':254}
codes_with_ratings = dict()

for c in codes:
    for r in [1,2,3,4]:
        val = codes[c] + r * 1000
        key = c + '/{}'.format(r)
        codes_with_ratings[key] = val

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

    raw.set_montage(montage)

    # zero-phase bandpass FIR filter with passband from 0.1 to 30 Hz
    raw.filter(0.1,30, method='fir', n_jobs=2)

    # extract events
    events = mne.find_events(raw)

    # create new events based on ratings #BT: we copy the original list of events, and this list gets modified by adding te 4 response triggers, it there is no item, append 0
    events_sequence = events[:,2]
    items = list()
    for idx, ev in enumerate(events[:,2]):
        if ev in [251, 252, 253, 254]:
            rating = events_sequence[idx + 4]
            events_sequence[idx] = ev + rating*1000
            items.append(events_sequence[idx+1])
        else:
            items.append(0)

    events[:,2] = events_sequence   #again replace it with the new column we created
    events[:,1] = items
    critical = events[events[:,1] != 0]  #because the non-critical got 0s
    # create epochs
    epochs = mne.Epochs(raw,events=critical, event_id=codes_with_ratings,
                        tmin=-0.2, tmax=1.2, # epochs go from -200 to +1200ms
                        detrend=1, # linear detrending
                        baseline=None, # No baselining
                        reject=peak_to_peak, # peak-to-peak rejections
                        flat=flatline, # flatline rejection
                        on_missing='ignore', # ignore missing events -- not all ratings-cond combos occu (BT: the standard is to give Error)
                        preload=True)
    epochs.drop_bad()
    # absolute threshold rejection
    bad_epoch_mask = abs_threshold(epochs.pick_types(eeg=True), threshold)
    epochs.drop(bad_epoch_mask,reason="absolute threshold")

    df = retrieve(epochs, wins,items=epochs.events[:,1])  #BT: items are in the event structure
    df.to_csv(csvout)

    for c in codes:
        erp = epochs[c].average()
        ssavg[c].append(erp)

#BT: because he kept getting errors
    for c in codes_with_ratings:
        try:
            erp = epochs[c].average()
            ssavg[c].append(erp)
        except KeyError: # this combination doesn't occur
            pass


# visualizing grand averages
# non-overlapping 83% CIs of the ind. measures correspond to the 95% CI
# of the difference
longfit = {ssavg[c] for c in ssavg if 'long/fit' in c}
mne.viz.plot_compare_evokeds(ssavg,picks=[15],invert_y=True,ci=.83)
