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

from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime


from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA

from scatseisnet import ScatteringNetwork

# %% 
# parameters

working_dir = '.'

saving = True
dirpath_save = './tests' # or can be defined in loop to put outputs in different folders
os.makedirs(dirpath_save, exist_ok=True)

saving_figs = True
dirpath_savefigs = './tests' # or can be defined in loop to put figs in different folders
os.makedirs(dirpath_save, exist_ok=True)

## Waveform data filtering parameters
lower_corner = 20.0 # highpass, for now

## Scattering network parameters - defined below
# segment_duration_seconds = 5
# # sampling_rate_hertz = 500.0
# reduce_type = np.max
# layer1_octaves = 4
# layer1_resolution = 4
# layer1_quality = 1
# layer2_octaves=5
# layer2_resolution=2
# layer2_quality=3

## FastICA parameter - defined below
# n_ICA_components=20

## clustering parameters - defined below
# N_CLUSTERS = 20 #k means


# %%
##### Load data to test on - local
import glob
from obspy.core import Stream, read

station='SE63'
tstart = datetime(2018,5,20,12, tzinfo=timezone.utc)
tstop = datetime(2018,5,20,13, tzinfo=timezone.utc)

stream_file = os.path.join(dirpath_save,"GL_scattering_stream_short.mseed")
if glob.glob(stream_file):
    stream = read(stream_file)
else:
    # Load data
    mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*140') # should be57 maybw ***
    stream = Stream()
    for f in mseed_files:
        stream += read(f)
    # mseed_files = glob.glob(f'/data/stor/basic_data/seismic_data/day_vols/MoVE/{station}/*141')
    # for f in mseed_files:
    #     stream += read(f)
    # trim to a shorter time period for processing, if desired
    stream.trim(stream[0].stats.starttime + 12*60*60, stream[0].stats.starttime + 13*60*60 ) # cut to an hour to start

    # merge, detrend, filter
    stream.merge(method=1)
    stream.detrend("linear")
    stream.filter(type="highpass", freq=lower_corner)

    # Downsample to 50 Hz
    # stream.decimate(factor=10) # 500 sps 
    # stream.decimate(factor=4) # 200 sps 
    # SE63 - don't need to downsample
    print(stream)
    stream.plot(rasterized=True)
    # Write out
    stream.write(stream_file, format="MSEED") # the long one still written out
for tr in stream:
    print(tr)
stream.plot()
sampling_rate_hertz = stream[0].stats.sampling_rate
# %% 
##### Load data to test on - from IRIS
# # Connect to the IRIS datacenter
# client = Client("IRIS")

# # Collect waveforms from the datacenter
# stream = client.get_waveforms(
#     network="YH",
#     station="DC08",
#     location="*",
#     channel="*",
#     starttime=UTCDateTime("2012-07-25T00:00"),
#     endtime=UTCDateTime("2012-07-26T00:00"),
# )



# %%
##### Specific to my dataset: load icequake rates from STA/LTA
sta_lta_detec_file = '/data/stor/proj/westGL_icequakes/IQ_occurrence/triggers/SE63.bandpass.20.100_0.1-2_6-5.triggers.ALL.csv'
all_stalta_detecs = pd.read_csv(sta_lta_detec_file)
all_stalta_detecs = all_detecs.assign(tON_dt2 = pd.to_datetime(all_detecs.tON_dt, utc=True))
mask = (all_stalta_detecs.tON_dt2 >= tstart) & (all_stalta_detecs.tON_dt2 <= tstop)
sta_lta_detecs = all_stalta_detecs[mask] ## detection rate just from STA/LTA

# hist of sta_lta_detecs
bin_width=15 # seconds
nbins = ((times[-1] + timedelta(seconds=segment_duration)) - times[0]).seconds/bin_width
bin_edges = np.arange(UTCDateTime(tstart).timestamp, UTCDateTime(tstop).timestamp+segment_duration_seconds, bin_width)
bin_edges_dt = [UTCDateTime(t).datetime.replace(tzinfo=timezone.utc) for t in bin_edges]
bin_centers_dt = [UTCDateTime(t+bin_width/2).datetime.replace(tzinfo=timezone.utc) for t in bin_edges[:-1]]

detec_times = [UTCDateTime(t) for t in pd.to_datetime(sta_lta_detecs.tON_dt)] #reformat into datetime; shouldn't have to, but its buggy otherwise
hist_stalta, _ = np.histogram(detec_times, bins=bin_edges_dt)

# %%
##### Specific to my dataset: load icequake rates from cross correlation, final families
f_detecs = "/data/stor/proj/westGL_icequakes/IQ_occurrence/clusters/chunksize_500/SE63/FINAL_detecs_SE63.csv"
all_corr_detecs = pd.read_csv(f_detecs)
all_corr_detecs = all_corr_detecs.assign(detec_time_dt = pd.to_datetime(all_corr_detecs.detec_time, utc=True))
mask = (all_corr_detecs.detec_time_dt >= tstart) & (all_corr_detecs.detec_time_dt <= tstop)
corr_detecs = all_corr_detecs[mask] ## detection rate just from STA/LTA

# %%
# detecs of repeaters on this day

# %%
##### List of different parameter sets to test
# order: segment_duration, reduce_type,
#        layer1_octaves, layer1_resolution, layer1_quality, 
#        layer2_octaves, layer2_resolution, layer2_quality, 
#        n_ICA_components, N_CLUSTERS
parameters_to_test = [ 
    [15,np.max,    4,4,1,   5,2,3,   20, 20], # 10 and 15 sec segment duration seems too long at 20/20 - not a lot of structure
    [10,np.max,    4,4,1,   5,2,3,   20, 20],
    [5,np.max,    4,4,1,   5,2,3,   20, 20], # 2 no cluster that obv resembles purple repeaters, but def lots of icequake clusters, 
    [2.5,np.max,    4,4,1,   5,2,3,   20, 20], # 3
    [1.5,np.max,    4,4,1,   5,2,3,   20, 20], # 4  cluster 3 looks like purple repeaters (sort of), def lots of icequake clusters, much more chaotic time series
    [15,np.max,    4,4,1,   5,2,3,   10, 10], # 5
    [10,np.max,    4,4,1,   5,2,3,   10, 10], # 6 cluster 4 closely resembles purple repeaters
    [5,np.max,    4,4,1,   5,2,3,   10, 10], # 7 cluster 4 again might resemble purple repeaters
    [2.5,np.max,    4,4,1,   5,2,3,   10, 10], # 8 cluster 1 really closely resembles repeaters, and cluster 4 from test 7
    [1.5,np.max,    4,4,1,   5,2,3,   10, 10], # starts losing structure again; maybe cluster 9 resembles purple repeaters
    ## Either 5 or 2.5 sec window works best; 20 components and 20 clusters gives more unique clusters that could be repeating iceuakes
    
    # test average versus max
    [5,np.average,    4,4,1,   5,2,3,   20, 20], # 10 - compare with 2 - some icequake fams; more equivocal difference  than for smaller # components and clusters (12, 13)
    [2.5,np.average,    4,4,1,   5,2,3,   20, 20], # 11 - compare with 3 - some icequake fams for 11; more equivocal difference with the np.max version
    [5,np.average,    4,4,1,   5,2,3,   10, 10],  # 12 (av) - compare with 7; again doesn't get as many clusters that are icequakes
    [2.5,np.average,    4,4,1,   5,2,3,   10, 10], # 13 (avg) gets low frequency thing nicely; compare with 8 (max) - 13 gets lf thing much better, but has few things that seem likely to be icequakes
    # np.average definitely damps the likelihood that multiple clusters will be icequake dominated
    # definitely also makes more likely that the low frequency thing gets its own cluster
    
    # test ncluster vs n feature
    [5,np.max,    4,4,1,   5,2,3,   20, 10], # 14 weirdly only one of these clusters looks like iceuakes
    [5,np.max,    4,4,1,   5,2,3,   20, 5], # 15 also here, none look like icequakes, but does get the interesting cluster that seems to go away early in the hour
    [5,np.max,    4,4,1,   5,2,3,   10, 6], # 16 Does seem to put all the icequakes in a single cluster, separate from other stuff - not super complete though
    [5,np.max,    4,4,1,   5,2,3,   10, 3], # 17
    # for pulling out icequakes, more features and more clusters seem to work better <<
    
    # test ncluster vs n feature
    ## OLD notes for when these were np.average(it really just didn't work)
    # [5,np.average,    4,4,1,   5,2,3,   20, 10], # not great for icequakes, but does get interesting amount of structure in clusters
    # [5,np.average,    4,4,1,   5,2,3,   20, 5], # also no diversity of clusters; doesnt get low frequency thing; does show change from early to later in time series which is interesting
    # [5,np.average,    4,4,1,   5,2,3,   10, 6], # not enough diversity of clusters to be useful for icequakes; def does get whatever the low frequency thing is
    # [5,np.average,    4,4,1,   5,2,3,   10, 3], # not useful; too few clusters
    # # averaging doesn't seem to get icequake clusters as well
    
    #segm,pooli      l1       l2      clustering
    
]

# Things to explore
# filter corner
# nfeature, ncluster
# np.max versus np.mean
# filterbanks


# %% 
##### Loop over tests of different filterbanks and clustering parameters
for testN, params in enumerate(parameters_to_test):
    print(testN)
    if testN > 13:
        ## %%
        ##### Pull out variables from list
        segment_duration_seconds = params[0]
        reduce_type = params[1]
        layer1_octaves = params[2]
        layer1_resolution = params[3]
        layer1_quality = params[4]
        layer2_octaves = params[5]
        layer2_resolution = params[6]
        layer2_quality = params[7]
        n_ICA_components = params[8]
        N_CLUSTERS = params[9]
        
        # dirpath_save = os.path.join(working_dir, f"test{testN:02}") # defined below
        # os.makedirs(dirpath_save, exist_ok=True)

        # if saving_figs:
        #     dirpath_savefigs = os.path.join(working_dir, f"test{testN:02}")
        #     os.makedirs(dirpath_save, exist_ok=True)
            
        # %%
        ##### Create scattering network
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

        if saving:
            # Save the scattering network with Pickle
            filepath_save = os.path.join(dirpath_save, f"GL_scattering_network_{testN}.pickle")
            with open(filepath_save, "wb") as file_save:
                pickle.dump(network, file_save, protocol=pickle.HIGHEST_PROTOCOL)

        # %%
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
            ax[1].set_xscale("log")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].grid()

            fig.suptitle(f"test {testN}\nlayer {ii}, seg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}")
            if saving_figs:
                filepath_save = os.path.join(dirpath_savefigs, f"GL_scattering_network_{testN}.png")
                fig.savefig(filepath_save, dpi=300)

                
        # %%
        ##### Chunk Seismograms
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
        # %%
        ##### Scattering transformation
        scattering_coefficients = network.transform(segments[:-1], reduce_type=reduce_type) 

        if saving:
            np.savez(
                os.path.join(dirpath_save, f"GL_scattering_coefficients_{testN}.npz"),
                order_1=scattering_coefficients[0],
                order_2=scattering_coefficients[1],
                times=timestamps,
            )

        # %%
        ##### Visualize scattering transform of a channel
        # Extract channel by component

        component = 'E' # just last character of chan code
        chans = [tr.stats.component == component for tr in stream]
        channel_id = np.arange(len(chans))[chans][0]
        trace = stream.select(component=component)[0]
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
        order_1 = scattering_coefficients[0]
        order_2 = scattering_coefficients[0]
        times = timestamps

        order_1 = order_1.reshape(order_1.shape[0], -1)
        order_2 = order_2.reshape(order_2.shape[0], -1)
        scattering_coefficients_reshaped = np.hstack((order_1, order_2))

        # transform into log
        scattering_coefficients_reshaped = np.log(scattering_coefficients_reshaped)

        # print info about shape
        n_times, n_coeff = scattering_coefficients_reshaped.shape
        print("Collected {} samples of {} dimensions each.".format(n_times, n_coeff))

        # %%
        ##### Reduction via fastICA
        ## THIS IS ALSO SOMETHING YOU CAN TRY CHANGING
        model = FastICA(n_components=n_ICA_components, whiten="unit-variance")
        features = model.fit_transform(scattering_coefficients_reshaped)

        if saving:
            # Save the features
            np.savez(
                os.path.join(dirpath_save, f"GL_independent_components_{testN}.npz"),
                features=features,
                times=times,
            )

            # Save the dimension reduction model
            with open(os.path.join(dirpath_save, f"GL_dimension_model_{testN}.pickle"), "wb") as pickle_file:
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
        fig.suptitle(f"test {testN}\nseg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}")

        # Show
        plt.show()

        if saving_figs:
            filepath_save = os.path.join(dirpath_savefigs, f"GL_features_{n_ICA_components}_{testN}.png")
            fig.savefig(filepath_save, dpi=300)


        # %%
        ##### DID NOT INCLUDE THE RECONSTRUCTION STEP

        # %%
        ##### Perform clustering
        # k-means clustering
        model = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=4)
        model.fit(features)

        # Predict cluster for each sample
        predictions = model.predict(features)

        # %% 
        ##### Cluster-wise detection rate
        SMOOTH_KERNEL = 20

        # Convert predictions to one-hot encoding
        one_hot = np.zeros((len(times[:-1]), N_CLUSTERS + 1))
        one_hot[np.arange(len(times[:-1])), predictions] = 1

        # Plot the results
        gs = gridspec.GridSpec(7,1)
        # fig, ax = plt.subplots(figsize=(6, 6))
        fig = plt.figure()
        ax = fig.add_subplot(gs[:5, :])

        # Plot each cluster as a separate line
        # times_tmp = [UTCDateTime(t).datetime for t in times] # same format as times below
        for i in range(N_CLUSTERS):

            # Obtain the detection rate by convolving with a boxcar kernel
            detection_rate = np.convolve(one_hot[:, i], np.ones(SMOOTH_KERNEL), mode="same") / SMOOTH_KERNEL

            # Plot the detection rate
            ax.plot(times[:-1], one_hot[:, i] + i, alpha=0.5)
            ax.plot(times[:-1], detection_rate + i, color="black")
        ax.set_xlim([tstart,tstop])

        # Labels
        ax.set_xlabel("Time")
        ax.set_ylabel("Cluster index")
        fig.suptitle(f"test {testN}\nseg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}, n_ICA_comp: {n_ICA_components}")

        ### Plot sta/lta detection rate
        # fig, ax = plt.subplots(figsize=(6, 6))
        ax = fig.add_subplot(gs[5,:])
        ax.bar(bin_centers_dt, hist_stalta, width=timedelta(seconds=bin_width), color='k')
        ax.set_xlabel("Time")
        ax.set_ylabel("# sta/lta\detecs")
        ax.set_xlim([tstart,tstop])

        ### Plot repeater detection rate, if more than 3/hr
        ax = fig.add_subplot(gs[6,:])
        many_colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
        # many_colors = many_colors[::2]
        for ix_fam, name in enumerate(corr_detecs.template_id.unique()):
            mask = corr_detecs.template_id == name
            df_tmp = corr_detecs[mask]
            if len(df_tmp) > 3:
                # print(name, len(df_tmp)) 
                detec_times = [UTCDateTime(t) for t in pd.to_datetime(df_tmp.detec_time_dt, utc=True)] 
                amps = df_tmp.maxamp_anychan.values
                for ix_detec, t in enumerate(detec_times):
                    x = [t,t,]
                    y = [0,np.log10(amps[ix_detec])]
                    ax.plot(x,y,'-', linewidth=0.5, color=many_colors[ix_fam])
                    ax.plot(x[1],np.log10(amps[ix_detec]),'*', color=many_colors[ix_fam])
        ax.set_ylim(1.45, 2.75)
        ax.set_xlabel("Time")
        ax.set_ylabel("log10 maxamp")
        ax.set_xlim([tstart,tstop])

        # plt.show()

        if saving_figs:
            filepath_save = os.path.join(dirpath_savefigs, f"GL_clusters_{n_ICA_components}_{N_CLUSTERS}_{testN}.png")
            fig.savefig(filepath_save, dpi=300)




        # %%
        ##### Visualize detections: extract detections
        N_WAVEFORMS = 5
        plot_component = 'Z'

        # re-load features  - changes something about formatting 
        with np.load(os.path.join(dirpath_save, f"GL_independent_components_{testN}.npz"), allow_pickle=True) as data:
            features = data["features"]
            times = data["times"]

        # # Read the stream
        # stream = obspy.read("GL_scattering_stream_short.mseed").select(channel="HHZ")
        waveform_duration = network.bins / network.sampling_rate

        ## for dealing with missing segment #* THIS WAS A BUG GRACE KEPT FINDING, this was the fix
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
                time = UTCDateTime(time)
                trace = stream.slice(time, time + waveform_duration).select(component=plot_component)[0].copy() 
                traces.append(trace)
            waveforms.append(traces)

        # %%
        ##### Sort of useful summary plot
        # Plot the results
        fig, ax = plt.subplots(N_WAVEFORMS, N_CLUSTERS, sharex=True, sharey=True)
        # Plot each cluster as a separate line
        for i, traces in enumerate(waveforms):
            ax[0, i].set_title(f"{i}", fontsize=8)#, rotation="vertical", fontsize=5)
            for j, trace in enumerate(traces):
                ax[j, i].plot(trace.times(), trace.data, rasterized=True, lw=0.6, color=f"C{i}")
                ax[j, i].set_axis_off()

        fig.suptitle(f"test {testN} seg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}, n_ICA_comp: {n_ICA_components}, {plot_component}", fontsize=10)

        plt.show()

        if saving_figs:
            filepath_save = os.path.join(dirpath_savefigs, f"GL_cluster_waveforms_{n_ICA_components}_{N_CLUSTERS}_{testN}.png")
            fig.savefig(filepath_save, dpi=300)




        # %%
        ##### Plot of 5 most typical events in each cluster
        from obspy.core import Stream
        # clustern = 4
        for i, lst in enumerate(waveforms):
            st = Stream()
            for tr in lst:
                tr.stats.starttime = lst[0].stats.starttime #just for plotting
                st.append(tr)
            fig, ax = plt.subplots(1, figsize = [7,4])
            st.plot(fig=fig, equal_scale=False)
            # fig.suptitle(f"cluster {i}")
            fig.suptitle(f"cluster {i} test {testN}seg dur: {segment_duration_seconds}, sps: {sampling_rate_hertz}, n_ICA_comp: {n_ICA_components}")

            if saving_figs:
                filepath_save = os.path.join(dirpath_savefigs, f"GL_cluster_waveforms{i:02}_{n_ICA_components}_{N_CLUSTERS}_{testN}.png")
                fig.savefig(filepath_save, dpi=300)
