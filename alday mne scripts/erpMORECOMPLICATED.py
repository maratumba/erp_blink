import pylab
import matplotlib.pyplot as plt
import mne
import numpy as np
import glob
import os.path
import pandas as pd
import warnings
import logging
mne.set_log_level('Warning')
mne.set_log_file('abraxas_erp.log',overwrite=True)
layout = mne.channels.read_layout("EEG1005.lay")
montage = mne.channels.read_montage(kind="standard_1020")

raw = {}
epochs = {}
evoked = {}

# event trigger and conditions
event_id = {'iien': 182,
            'iiev': 181,
            'iaen': 172,
            'iaev': 171,
            'aien': 162,
            'aiev': 161,
            'aaen': 152,
            'aaev': 151,
            'iipn': 142,
            'iipv': 141,
            'iapn': 132,
            'iapv': 131,
            'aipn': 122,
            'aipv': 121,
            'aapn': 112,
            'aapv': 111}

windows = {
        "p200": (150,250),
        "n400" : (300, 500),
        "n400-early" : (300, 400),
        "n400-late" : (400, 500),
        "p600" : (600, 800)
        }


for eeg in glob.glob('raw/*[0-9].vhdr'):
    subj = os.path.splitext(os.path.basename(eeg))[0]
    print subj,
    raw = mne.io.read_raw_brainvision(eeg,
                                      eog=("EOGre", "EOGli","EOGunre","EOGobre"),
                                      misc=("RerRef",),
                                      montage=montage,
                                      preload=True)
    raw.filter(0.16, 30,
               l_trans_bandwidth=0.1,
               h_trans_bandwidth=1,
               method='fir',
               n_jobs=2,
               filter_length='auto',
               phase='zero',
               fir_window='hamming')
    raw = mne.io.add_reference_channels(raw,"Ref")


    raw = mne.io.set_eeg_reference(raw,["RerRef","Ref"])[0]
    events = mne.find_events(raw)

    # extract events that had a correct response
    eventsq = events[:,2]
    useevents = np.zeros(len(eventsq),dtype=bool)
    items = np.zeros(len(eventsq),dtype=int)
    item = None

    for i,e in enumerate(eventsq):
        if 110 < e < 190:
            if e % 10 == 1:
                if eventsq[i+4] == 196:
                    useevents[i] = True;
                    items[i] = item;
            elif e % 10 == 2:
                if eventsq[i+3] == 196:
                    useevents[i] = True;
                    items[i] = item;
            else:
                pass
        elif e > 200:
            item = e
        else:
            pass

    events[:,1] = items
    events = events[useevents]

    # assign 999 to EOG events
    #eog_events = mne.preprocessing.find_eog_events(raw, 999)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False, misc=False)


    tmin, tmax = -0.2, 1.2
    baseline =  None
    reject = dict(eeg=70e-6,eog=100e-6)
    flat = dict(eeg=5e-6)
    epochs = mne.Epochs(raw, events,
                        event_id,
                        tmin, tmax,
                        baseline=baseline,
                        preload=True,
                        reject=reject,
                        flat=flat,
                        picks=picks,
                        detrend=0,
                        on_missing='warning')
    epochs.drop_bad()
    try:
        if len(epochs) < 10:
           print "EXCLUDED"
           continue
        else:
            epochs.save(os.path.join("data",subj+"-epo.fif.gz"))
    except IndexError:
        print "EXCLUDED"
        continue

    items = epochs.events[:,1]
    print len(epochs)
    df = epochs.to_data_frame(picks=None, index=['epoch','time'],scale_time=1e3)
    eeg_chs = [c for c in df.columns if c not in ('Ref', 'RerRef', 'EOGre','EOGli','EOGunre','EOGobre','condition')]
    factors = ['epoch','condition'] # the order is important here! otherwise the shortcut with items later won't  work
    sel = factors + eeg_chs
    df = df.reset_index()

    retrieve = []
    for w in windows:
        temp = df[ df.time >= windows[w][0] ]
        dfw = temp[ temp.time <= windows[w][1] ]
        dfw_mean = dfw[sel].groupby(factors).mean()
        dfw_mean["subj"] = subj
        dfw_mean["item"] = items
        dfw_mean["win"] = "{}..{}".format(*windows[w])
        dfw_mean["wname"] = w
        retrieve.append(dfw_mean)

    retrieve = pd.concat(retrieve)
    retrieve.to_csv(os.path.join("data",subj+".csv"))

    evoked[subj] = {}
    for e in event_id:
        with warnings.catch_warnings(record=True) as w:
            evoked[subj][e] = epochs[e].average()

        for ww in w:
            if "evoked object is empty" not in str(ww.message) and "Mean of empty slice" not in str(ww.message) :
               print ww

evoked_by_cond = {e:[evoked[s][e] for s in evoked] for e in event_id}
grand_average = {cond:mne.grand_average(evoked_by_cond[cond]) for cond in evoked_by_cond}

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open("cache.pickle","wb") as pfile:
    pickle.dump(evoked,pfile,-1)
    pickle.dump(evoked_by_cond,pfile,-1)
    pickle.dump(grand_average,pfile,-1)
