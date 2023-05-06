
'''
This is yoram's version of combining all the tutorial notebooks into one long workflow. 
Aim is to test it first on the tremor data from E57 and then look for tremor changes. '''


#%% PART I: SCATTERING NETWORK DESIGN ##############################################
####################################################################################



#%% 
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from scatseisnet import ScatteringNetwork

from matplotlib import dates as mdates

from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

plt.rcParams["date.converter"] = "concise"


#%% Select parameters
segment_duration_seconds = 500.0
sampling_rate_hertz = 50.0
samples_per_segment = int(segment_duration_seconds * sampling_rate_hertz)
bank_keyword_arguments = (
    {"octaves": 2, "resolution": 5, "quality": 1},
    {"octaves": 8, "resolution": 2, "quality": 4},
)

#%% create scatering network 
network = ScatteringNetwork(
    *bank_keyword_arguments,
    bins=samples_per_segment,
    sampling_rate=sampling_rate_hertz,
)

print(network)

#%% save the network 
dirpath_save = "./HC"

# Create directory to save the results
os.makedirs(dirpath_save, exist_ok=True)

# Save the scattering network with Pickle
filepath_save = os.path.join(dirpath_save, "GL_scattering_network_HC.pickle")
with open(filepath_save, "wb") as file_save:
    pickle.dump(network, file_save, protocol=pickle.HIGHEST_PROTOCOL)


#%% Visaualise filter banks 

# Loop over network layers
for bank in network.banks:

    # Create axes (left for temporal, right for spectral domain)
    fig, ax = plt.subplots(1, 2, sharey=True)

    # Show each wavelet
    for wavelet, spectrum, ratio in zip(
        bank.wavelets, bank.spectra, bank.ratios
    ):

        # Time domain
        ax[0].plot(bank.times, wavelet.real.get() + ratio, "C0")

        # Spectral domain (log of amplitude)
        ax[1].plot(bank.frequencies, np.log(np.abs(spectrum.get()) + 1) + ratio, "C0")

    # Limit view to three times the temporal width of largest wavelet
    width_max = 3 * bank.widths.max()

    # Labels
    ax[0].set_ylabel("Octaves (base 2 log)")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_xlim(-width_max, width_max)
    ax[0].grid()
    #ax[1].set_xscale("log")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].grid()

#%% PART II: LOAD and pre-process SEISMIC DATA ##########################################
#########################################################################################


import glob
from obspy.core import Stream, read

#%% load data 
station='SEHC'
    
mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*230') # should be57 maybw ***
stream = Stream()
for f in mseed_files:
    stream += read(f)
mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*231')
for f in mseed_files:
    stream += read(f)

print(stream)

#%% pre-process data
stream.merge(method=1)
stream.detrend("linear")
#stream.filter(type="highpass", freq=1.0)
stream.filter(type="bandpass",freqmin=3,freqmax=10)

# Downsample to 50 Hz
# stream.decimate(factor=10) # 500 sps 
stream.decimate(factor=4) # 200 sps 
print(stream)
stream.plot(rasterized=True)

# save stream for later use in plotting and investigating waveforms
stream.write("./HC/GL_scattering_stream_HC.mseed", format="MSEED")


#%% chunk data into segments of equal length 
# Extract segment length (from any layer)
segment_duration = network.bins / network.sampling_rate
overlap = 0.5

# Gather list for timestamps and segments
timestamps = list()
segments = list()

# Collect data and timestamps
for traces in stream.slide(segment_duration, segment_duration * overlap):
    timestamps.append(mdates.num2date(traces[0].times(type="matplotlib")[0]))
    segments.append(np.array([trace.data[:-1] for trace in traces]))


#%% PART III: SCATTERING TRANSFORMATION ##############################################
######################################################################################

#%% 
# reload the scattering network info 
network = pickle.load(open("./HC/GL_scattering_network_HC.pickle", "rb"))

# execute transformation 
scattering_coefficients = network.transform(segments[:-1], reduce_type=np.average)


# %% observe resut from single channel 
 # Extract the first channel
channel_id = 0
trace = stream[channel_id]
order_1 = np.log10(scattering_coefficients[0][:, channel_id, :].squeeze())
center_frequencies = network.banks[0].centers

# Create figure and axes
fig, ax = plt.subplots(2, sharex=True, dpi=300)

# Plot the waveform
ax[0].plot(trace.times("matplotlib"), trace.data, rasterized=True, lw=0.5)

# First-order scattering coefficients
ax[1].pcolormesh(timestamps[:-1], center_frequencies, order_1.T, rasterized=True)

# Axes labels
ax[1].set_yscale("log")
ax[0].set_ylabel("Counts")
ax[1].set_ylabel("Frequency (Hz)")

# Show
plt.show()

#%% save the scattering coefficients 
np.savez(
    "./HC/GL_scattering_coefficients_HC.npz",
    order_1=scattering_coefficients[0],
    order_2=scattering_coefficients[1],
    times=timestamps,
)


#%% PART IV: DIMENSIONALITY REDUCTION ####################################################
##########################################################################################

from sklearn.decomposition import FastICA

plt.rcParams["date.converter"] = "concise"


#%% re-load the scattering network 

# Load data from file
with np.load("./HC/GL_scattering_coefficients_HC.npz", allow_pickle=True) as data:
    order_1 = data["order_1"]
    order_2 = data["order_2"]
    times = data["times"]

# Reshape and stack scattering coefficients of all orders
order_1 = order_1.reshape(order_1.shape[0], -1)
order_2 = order_2.reshape(order_2.shape[0], -1)
scattering_coefficients = np.hstack((order_1, order_2))

# transform into log
scattering_coefficients = np.log(scattering_coefficients)

# print info about shape
n_times, n_coeff = scattering_coefficients.shape
print("Collected {} samples of {} dimensions each.".format(n_times, n_coeff))

# %% Extract indepenedant features with FASTICA 
n_components=5
model = FastICA(n_components=n_components, whiten="unit-variance")
features = model.fit_transform(scattering_coefficients)

# Save the features
np.savez(
    "./HC/GL_independent_components_HC.npz",
    features=features,
    times=times,
)

# Save the dimension reduction model
with open("./HC/GL_dimension_model_HC.pickle", "wb") as pickle_file:
    pickle.dump(
        model,
        pickle_file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

#%% plot the features as time series
# # Normalize features for display
features_normalized = features / np.abs(features).max(axis=0)

# Figure instance
fig = plt.figure(dpi=200, figsize=[6,6])
ax = plt.axes()

# Plot features
ax.plot(times[:-1], features_normalized + np.arange(features.shape[1]), rasterized=True)

# Labels
ax.set_ylabel("Feature index")
ax.set_xlabel("Date and time")

# Show
plt.show()


#%% PART V: CLUSTER THE FEATURES 

from scipy import signal
from sklearn.cluster import KMeans

plt.rcParams["date.converter"] = "concise"

#%  load features and network 

# Load features and datetimes from file
with np.load("./HC/GL_independent_components_HC.npz", allow_pickle=True) as data:
    features = data["features"]
    times = data["times"]

# Load network
network = pickle.load(open("./HC/GL_scattering_network_HC.pickle", "rb"))


#% apply K-means clustering 
N_CLUSTERS = 5

# Perform clustering
model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=4)
model.fit(features)

# Predict cluster for each sample
predictions = model.predict(features)

#% VISUALISE DETECTIONS RATE 
SMOOTH_KERNEL = 5

# Convert predictions to one-hot encoding
one_hot = np.zeros((len(times[:-1]), N_CLUSTERS + 1))
one_hot[np.arange(len(times[:-1])), predictions] = 1

# Plot the results
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each cluster as a separate line
for i in range(N_CLUSTERS):

    # Obtain the detection rate by convolving with a boxcar kernel
    detection_rate = np.convolve(one_hot[:, i], np.ones(SMOOTH_KERNEL), mode="same") / SMOOTH_KERNEL

    # Plot the detection rate
    ax.plot(times[:-1], one_hot[:, i] + i, alpha=0.5)
    ax.plot(times[:-1], detection_rate + i, color="black")

# Labels
ax.set_xlabel("Time")
ax.set_ylabel("Cluster index")

plt.show()


#% GET WAVEFORMS 
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
            
            
# %
# Plot the results
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each cluster as a separate line
for i in range(N_CLUSTERS):
    ax.plot(f1, ppd_conv[i,:],label='Cluster '+str(i))
# Labels
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Median Amplitude")
ax.set_xlim(2.5,11)
ax.set_ylim(100,12000)
ax.set_yscale('log')
plt.legend()
plt.show()
# %%


#%% 

from obspy.core.inventory.inventory import read_inventory
inv = read_inventory('/data/stor/basic_data/seismic_data/day_vols/MoVE/Resp/MoVE_station.xml')
print(inv)
# %%
