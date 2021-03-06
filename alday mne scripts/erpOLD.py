bplist00�_WebMainResource�	
_WebResourceMIMEType_WebResourceTextEncodingName_WebResourceFrameName^WebResourceURL_WebResourceDataZtext/plainUUTF-8P_0https://gitlab.com/palday/odie/raw/master/erp.pyOc<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">codes = {'long/fit':251, 'long/no-fit':252, 'short/fit':253, 'short/no-fit':254}

ssavg = {}
for c in codes:
    ssavg[c] = list()

for i in range(5):
    subjfile = 'garfield00{:02d}.vhdr'.format(i+1)
    raw = mne.io.read_raw_brainvision(subjfile, preload=True)
    mne.rename_channels(raw.info, {'TP9':'A1', 'TP10':'A2', 'PO9':'IO1', 'PO10':'SO1', 'T7':'LO1', 'T8':'LO2'})
    raw.set_eeg_reference(['A1', 'A2'])
    raw.set_channel_types({'IO1':'eog', 'SO1':'eog', 'LO1':'eog', 'LO2':'eog'})
    raw.filter(0.1,30, method='fir')
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw,events=events,event_id=codes,tmin=-0.2,tmax=1.2, detrend=1,baseline=None)

    for c in codes:
        erp = epochs[c].average()
        ssavg[c].append(erp)
        

</pre></body></html>    ( > \ s � � � � � �                           @