''' This script pulls the cluster timeseries and tries to compare them with other data sources.'''

#%% load data 

from pandas import read_csv
import matplotlib.pyplot as plt
import pickle 
import numpy as np 
import datetime
sg_times=[]
sg = read_csv('/data/stor/proj/IQ_classes/scripts/seydoux_approach/yor_version/scatseisnet/GL_example/HC/data/SBPI_QIR.csv')
for t in range(len(sg["Timestamp (UTC)"])):
    
    sg_times.append(datetime.datetime.strptime(sg["Timestamp (UTC)"][t],'%Y-%m-%d %H:%M:%S'))

sg_times = np.array(sg_times)    
sg_Q = np.array(sg['Discharge smoothed (m^3/s)'])


# %% load the feature timesries and cluster them 

from scipy import signal
from sklearn.cluster import KMeans

plt.rcParams["date.converter"] = "concise"

#%%  load features and network 

# Load features and datetimes from file
with np.load("./HC/GL_independent_components_HC.npz", allow_pickle=True) as data:
    features = data["features"]
    times = data["times"]

# Load network
network = pickle.load(open("./HC/GL_scattering_network_HC.pickle", "rb"))


#%% apply K-means clustering 
N_CLUSTERS = 5

# Perform clustering
model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=4)
model.fit(features)

# Predict cluster for each sample
predictions = model.predict(features)

#%% VISUALISE DETECTIONS RATE 
SMOOTH_KERNEL = 5

# Convert predictions to one-hot encoding
one_hot = np.zeros((len(times[:-1]), N_CLUSTERS + 1))
one_hot[np.arange(len(times[:-1])), predictions] = 1

# Plot the results
fig, (ax,ax2) = plt.subplots(2,1,figsize=(14, 8))

# Plot each cluster as a separate line
for i in range(N_CLUSTERS):

    # Obtain the detection rate by convolving with a boxcar kernel
    detection_rate = np.convolve(one_hot[:, i], np.ones(SMOOTH_KERNEL), mode="same") / SMOOTH_KERNEL

    # Plot the detection rate
    ax.plot(times[:-1], one_hot[:, i] + i, alpha=0.5)
    ax.plot(times[:-1], detection_rate + i, color="black")

ax2.plot(sg_times,sg_Q)
ax2.set_xlim(datetime.datetime(2018,8,18,00,00,00),datetime.datetime(2018,8,20,00,00,00))
ax.set_xlim(datetime.datetime(2018,8,18,00,00,00),datetime.datetime(2018,8,20,00,00,00))
# Labels
ax2.set_xlabel("Time")
ax2.set_ylabel('Discharge (m3/s)')
ax.set_ylabel("Cluster index")

plt.show()



# %%

import obspy 


N_WAVEFORMS = 3

# Read the stream
stream = obspy.read("./HC/GL_scattering_stream_HC.mseed").select(channel="HHZ")
waveform_duration = network.bins / network.sampling_rate

## for dealing with missing segment
times0 = times[:-1]

# Extract waveforms
waveforms = list()
for cluster in np.unique(predictions):

    # Calculate the distance of each sample to the cluster mean
    mean = np.mean(features[predictions == cluster], axis=0)
    distance = np.linalg.norm(features[predictions == cluster] - mean, axis=1)
    closest = times0[predictions == cluster][distance.argsort()[:5]]
    #closest = times[predictions == cluster][distance.argsort()[:5]]

    # Collect closest waveforms in a list
    traces = list()
    for time in closest[:N_WAVEFORMS]:
        time = obspy.UTCDateTime(time)
        trace = stream.slice(time, time + waveform_duration)[0].copy() 
        traces.append(trace)
    waveforms.append(traces)


# % plot waveforms 

# Plot the results
fig, ax = plt.subplots(N_WAVEFORMS, N_CLUSTERS, sharex=True, sharey=True)

# Plot each cluster as a separate line
for i, traces in enumerate(waveforms):
    ax[0, i].set_title(f"Cluster {i}", rotation="vertical")
    for j, trace in enumerate(traces):
        ax[j, i].plot(trace.times(), trace.data, rasterized=True, lw=0.6, color=f"C{i}")
        ax[j, i].set_axis_off()

# Show
plt.show()

# % extract all the waveforms in each cluster and make median psds for them  

# Extract waveforms
waveforms = list()
for cluster in np.unique(predictions):

    # Calculate the distance of each sample to the cluster mean
    mean = np.mean(features[predictions == cluster], axis=0)
    distance = np.linalg.norm(features[predictions == cluster] - mean, axis=1)
    closest = times0[predictions == cluster][distance.argsort()[:]]
    #closest = times[predictions == cluster][distance.argsort()[:5]]

    # Collect closest waveforms in a list
    traces = list()
    for time in closest[:]:
        time = obspy.UTCDateTime(time)
        trace = stream.slice(time, time + waveform_duration)[0].copy() 
        traces.append(trace)
    waveforms.append(traces)


#%
from scipy.signal import periodogram 
sampling_rate_hertz = 50 
kernel_size = 150
kernel = np.ones(kernel_size) / kernel_size

print(waveforms[0][0])
# do a first one to get the array size 
f1, ppd1 = periodogram(waveforms[0][0],fs=sampling_rate_hertz)

ppd_conv = np.zeros((len(waveforms),len(f1)))

#%
for j in range(len(waveforms)):
    
    ppd = np.zeros((len(waveforms[j]),len(f1)))
    for i in range(len(waveforms[j])):
        f,ppd[i,:] = periodogram(waveforms[j][i],fs=sampling_rate_hertz)
        ppd_med = np.median(ppd,axis=0)
        ppd_conv[j,:] = np.convolve(ppd_med, kernel, mode='same')
    #plt.plot(f,ppd_conv[j,:])
            
            
#%%
# Plot the results
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each cluster as a separate line
for i in range(N_CLUSTERS):
    ax.plot(f1, ppd_conv[i,:],label='Cluster '+str(i))
# Labels
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Median Amplitude")
ax.set_xlim(2.5,11)
ax.set_ylim(100,3000)
ax.set_yscale('log')
plt.legend()
plt.show()

# %%
