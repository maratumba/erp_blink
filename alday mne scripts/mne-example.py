import mne
import numpy as np
import pandas as pd
import bambi 
import seaborn
import statsmodels
import glob
mne.set_log_level('INFO')

def abs_threshold(epochs, threshold):
    ''' Compute boolean mask for dropping epochs based on absolute voltage threshold'''

    data = epochs.pick_types(eeg=True,misc=False,stim=False).get_data()
    # channels and times are last two dimension in MNE ndarrays,
    # and we collapse across them to get a (n_epochs,) shaped array
    rej = np.any( np.abs(data) > threshold, axis=(-1,-2))

    return rej

def retrieve(epochs, windows, subj=None, items=None):
    df = epochs.to_data_frame(picks=None, index=['epoch','time'],scale_time=1e3)
    eeg_chs = [c for c in df.columns if c not in ('condition')]
    factors = ['epoch','condition'] # the order is important here! otherwise the shortcut with items later won't  work
    sel = factors + eeg_chs
    df = df.reset_index()

    retrieve = []
    for w in windows:
        temp = df[ df.time >= windows[w][0] ]
        dfw = temp[ temp.time <= windows[w][1] ]
        dfw_mean = dfw[sel].groupby(factors).mean()
        if subj:
            dfw_mean["subj"] = subj
        if items:
            dfw_mean["item"] = items
        dfw_mean["win"] = "{}..{}".format(*windows[w])
        dfw_mean["wname"] = w
        retrieve.append(dfw_mean)

    retrieve = pd.concat(retrieve)
    return retrieve

def ss_preproc(set_file, event_id, windows):
    raw = mne.io.read_raw_eeglab(set_file,eog=('EOGV','EOGH'),preload=True)
    # I suspect that EOGV and EOGH are bipolar recordings but they aren't marked as such,
    # so they're being re-referenced here too, which is a tad problematic
    raw.set_eeg_reference(['A1','A2'])
    raw = raw.copy().filter(0.3,30,
                            phase='zero',
                            method='fir',
                            l_trans_bandwidth='auto',
                            h_trans_bandwidth='auto')
    events = mne.find_events(raw)
    
    # start of each epoch in seconds relative to time-locking trigger
    tmin = -0.2  
    # end of each epoch in seconds  relative to time-locking trigger
    tmax = 1.2 
    # baseline interval in seconds. Use None for no baseline, or (None, None) for entire epoch
    baseline = None

    # We want EEG and EOG channels, but we dump channels previously marked as bad
    picks = mne.pick_types(raw.info, meg=False,eeg=True, eog=True, stim=False, exclude = 'bads')

    # we use rather general thresholds for the MNE built-in peak-to-peak method
    # because otherwise you throw out too much data.
    # to compensate for for this, we detect blinks separately and then apply an
    # absolute threshold to the epoched data

    # maximal peak-to-peak rejection amplitude
    # normally we would specify an EOG threshold, but due to the 
    # re-referenced bipolar thing (see above), this would reject all the 
    # epochs. The bink detection code below catches most eye events anyway
    reject = dict(eeg=150e-6) 
    # minimal peak-to-peak rejection amplitude 
    flat = dict(eeg=5e-6) 

    # catch obvious blinks, see the EOG artifacts example in the docs
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                      orig_time=raw.info['meas_date'])

    
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=tmin, tmax=tmax, 
                        proj=False, picks=picks, 
                        baseline=baseline, 
                        detrend=0, # DC offset
                        reject_by_annotation=True,
                        flat=flat, 
                        reject=reject, 
                        preload=True)

    # get rid of epochs that have any points exceeding a given absolute threshold
    # remember that MNE stores the data internally on the volt scale,
    # so we need the 1e-6 to get to the desired microvolt scale
    bad_epoch_mask = abs_threshold(epochs, 75e-6)

    epochs.drop(bad_epoch_mask,reason="absolute threshold")
    
    evokeds = {str(cond):epochs[str(cond)].average() for cond in event_id}
    wins = retrieve(epochs, windows, set_file)
    
    return evokeds, wins
    
    
event_id = {'a':304,'b':305}
windows = {
#    "prestimulus": (-200,0),
    "n400" : (300, 500),
#    "p600" : (600, 800)
    }
set_files = glob.glob('rumba/unproc/*.set')

ss_avg = dict() # new empty dict
ss_wins = dict()
# dict with empty list for each condition
ss_avg_by_cond = {e:list() for e in event_id} 
for filename in set_files:
    ss_avg[filename], ss_wins[filename] =  ss_preproc(filename,event_id=event_id,windows=windows)
        
    for cond in event_id:
        ss_avg_by_cond[cond].append(ss_avg[filename][cond])

# example evoked for determining channel indices
e = ss_avg_by_cond['a'][0]
cz_idx = mne.pick_channels(e.ch_names, ['CZ'])

mne.viz.plot_compare_evokeds(ss_avg_by_cond,picks=cz_idxs)

data = pd.concat(ss_wins)
data = data.reset_index(level=['condition'])

mod = bambi.Model(data)
results = mod.fit('CZ ~ condition', samples=5000)
results.summary(burnin=2000)
