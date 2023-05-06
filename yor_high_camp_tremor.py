# %%
# Load necessary modules
import os
import pickle

import matplotlib.pyplot as plt
plt.rcParams["date.converter"] = "concise"
from matplotlib import dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from datetime import timedelta

from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import obspy 


from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from scatseisnet import ScatteringNetwork

# %% 
# parameters

wdir = '/data/stor/proj/IQ_classes/scripts/seydoux_approach/yor_version/scatseisnet/GL_example/high_camp/'

saving = True
dirpath_save = wdir  # or can be defined in loop to put outputs in different folders
os.makedirs(dirpath_save, exist_ok=True)

saving_figs = True
dirpath_savefigs = wdir+'tests/' # or can be defined in loop to put figs in different folders
os.makedirs(dirpath_save, exist_ok=True)

## Waveform data filtering parameters
lower_corner = 20.0 # highpass, for now

# Scattering network parameters - defined below
segment_duration_seconds = 1000.0
sampling_rate_hertz = 50.0

layer1_octaves = 2
layer1_resolution = 5
layer1_quality = 1
layer2_octaves=8
layer2_resolution=2
layer2_quality=4

# reduce type 
reduce_type = np.average

# FastICA parameter - defined below
n_ICA_components=5

# clustering parameters - defined below
N_CLUSTERS = 5 #k means

N_WAVEFORMS = 3 

SMOOTH_KERNEL = 180

# %% 
saving = True
saving_figs = False

# %% Load data to test on - locally stored data
import glob
from obspy.core import Stream, read


stream_file = dirpath_save+"GL_scattering_stream_high_camp.mseed"

station='SE57'
    
mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*212') # should be57 maybw ***
stream = Stream()
for f in mseed_files:
    stream += read(f)
mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*213')
for f in mseed_files:
    stream += read(f)

print(stream)
stream.merge(method=1)
stream.detrend("linear")
stream.filter(type="bandpass",freqmin=3,freqmax=10)

# Downsample to 50 Hz
stream.decimate(factor=4) # used to be 200 sps 
print(stream)
stream.plot(rasterized=True)

stream.write(stream_file, format="MSEED")


# %% Create scattering network
samples_per_segment = int(segment_duration_seconds * sampling_rate_hertz)
bank_keyword_arguments = (
    {"octaves": layer1_octaves, "resolution": layer1_resolution, "quality": layer1_quality},
    {"octaves": layer2_octaves, "resolution": layer2_resolution, "quality": layer2_quality},
)

network = ScatteringNetwork(
    *bank_keyword_arguments,
    bins=samples_per_segment,
    sampling_rate=sampling_rate_hertz,
)
print(network)

# Save the scattering network with Pickle
filepath_save = dirpath_save+"GL_scattering_network_high_camp.pickle"
with open(filepath_save, "wb") as file_save:
    pickle.dump(network, file_save, protocol=pickle.HIGHEST_PROTOCOL)

        ##### Look at filter banks
        # Loop over network layers
for ii, bank in enumerate(network.banks):

    # Create axes (left for temporal, right for spectral domain)
    fig, ax = plt.subplots(1, 2, sharey=True)

    # Show each wavelet
    for wavelet, spectrum, ratio in zip(
        bank.wavelets, bank.spectra, bank.ratios
    ):

        # Time domain
        ax[0].plot(bank.times, wavelet.get().real + ratio, "C0")

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

    fig.suptitle(f"test_layer {ii}, seg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}")
    if saving_figs:
        filepath_save = [dirpath_savefigs+"GL_scattering_network_layer_{ii}_.png"]
        fig.savefig(filepath_save, dpi=300)



##% reload network 

network = pickle.load(open(dirpath_save+"GL_scattering_network_high_camp.pickle", "rb"))
                
# %% Chunk Seismograms
## ** There is probably a way to make this much faster with *broadcasting* - haven't coded it yet
# Extract segment length (from any layer)
segment_duration = network.bins / network.sampling_rate
print('segment_duration', segment_duration)

overlap = 0.5

# Gather list for timestamps and segments
timestamps = list()
segments = list()

# Collect data and timestamps
for traces in stream.slide(segment_duration, segment_duration * overlap):
    timestamps.append(mdates.num2date(traces[0].times(type="matplotlib")[0]))
    segments.append(np.array([trace.data[:-1] for trace in traces]))
    
# %% Scattering transformation
scattering_coefficients = network.transform(segments[:-1], reduce_type=reduce_type) 

if saving:
    np.savez(
        dirpath_save+"GL_scattering_coefficients_high_camp.npz",
        order_1=scattering_coefficients[0],
        order_2=scattering_coefficients[1],
        times=timestamps,
    )

#%% 
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


# %%
##### Reshape and stack scattering coefficients of all orders
# order_1 = scattering_coefficients[0]
# order_2 = scattering_coefficients[0]
# times = timestamps

# order_1 = order_1.reshape(order_1.shape[0], -1)
# order_2 = order_2.reshape(order_2.shape[0], -1)
# scattering_coefficients_reshaped = np.hstack((order_1, order_2))

# # transform into log
# scattering_coefficients_reshaped = np.log(scattering_coefficients_reshaped)

# # print info about shape
# n_times, n_coeff = scattering_coefficients_reshaped.shape
# print("Collected {} samples of {} dimensions each.".format(n_times, n_coeff))

#%% Load data from file
with np.load(dirpath_save+"GL_scattering_coefficients_high_camp.npz", allow_pickle=True) as data:
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

# %% Reduction via fastICA
model = FastICA(n_components=n_ICA_components, whiten="unit-variance")
features = model.fit_transform(scattering_coefficients)

if saving:
    # Save the features
    np.savez(
        dirpath_save+"GL_independent_components_high_camp.npz",
        features=features,
        times=times,
    )

    # Save the dimension reduction model
    with open(dirpath_save+"/GL_dimension_model_high_camp.pickle", "wb") as pickle_file:
        pickle.dump(
            model,
            pickle_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

# %%
##### Visualize the features from the reduction
# Normalize features for display
features_normalized = features / np.abs(features).max(axis=0)

# Figure instance
fig = plt.figure(dpi=200, figsize=[6,6])
ax = plt.axes()

# Plot features
ax.plot(times[:-1], features_normalized + np.arange(features.shape[1]), rasterized=True)

# Labels
ax.set_ylabel("Feature index")
ax.set_xlabel("Date and time")
fig.suptitle(f"segm duration: {segment_duration_seconds}, sps: {sampling_rate_hertz}")

# Show
plt.show()

if saving_figs:
    filepath_save = os.path.join(dirpath_savefigs, f"GL_features_{n_ICA_components}.png")
    fig.savefig(filepath_save, dpi=300)


# %% DID NOT INCLUDE THE RECONSTRUCTION STEP

# %% CLUSTERING 

# Perform clustering
model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=4)
model.fit(features)

# Predict cluster for each sample
predictions = model.predict(features)

#%% 
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

#%% visualise time evolution of clusters 

# Convert predictions to one-hot encoding
one_hot = np.zeros((len(times[:-1]), N_CLUSTERS + 1))
one_hot[np.arange(len(times[:-1])), predictions] = 1

# Plot the results
fig, ax = plt.subplots(figsize=(14, 8))
SMOOTH_KERNEL = 180
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
# %%

# Read the stream
stream = obspy.read(dirpath_save+"GL_scattering_stream_high_camp.mseed").select(channel="HHZ")
waveform_duration = network.bins / network.sampling_rate

## for dealing with missing segment
times0 = np.array(times[:-1])

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
# %% Plot the results
fig, ax = plt.subplots(N_WAVEFORMS, N_CLUSTERS, sharex=True, sharey=True)

# Plot each cluster as a separate line
for i, traces in enumerate(waveforms):
    ax[0, i].set_title(f"Cluster {i}", rotation="vertical")
    for j, trace in enumerate(traces):
        ax[j, i].plot(trace.times(), trace.data, rasterized=True, lw=0.6, color=f"C{i}")
        ax[j, i].set_axis_off()

# Show
plt.show()
# %%
