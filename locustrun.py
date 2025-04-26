from urllib.request import urlretrieve # Python 3
# from urllib import urlretrieve # Python 2
data_names = ['Locust_' + str(i) + '.dat.gz' for i in range(1,5)]
data_src = ['http://xtof.disque.math.cnrs.fr/data/' + n
            for n in data_names]
data_src

import os
print(os.getcwd() + "\n")

pip install numpy
pip install matplotlib scipy
import sorting_with_python as swp

import numpy as np
import matplotlib.pylab as plt
import math as m
plt.ion()

# Create a list with the file names
data_files_names = ['C:/Users/khushbu_fegade/Downloads/Locust_' + str(i) + '.dat' for i in range(1,5)]
print(data_files_names)
# Get the lenght of the data in the files
data_len = np.unique(list(map(len, map(lambda n:
                                       np.fromfile(n,np.double),
                                       data_files_names))))[0]
# Load the data in a list of numpy arrays
data = [np.fromfile(n,np.double) for n in data_files_names]

def get_correlation_df(site):
    return pd.DataFrame(zip(site[0], site[1], site[2], site[3]),  columns=['site1', 'site2', 'site3', 'site4'])

import pandas as pd

df = get_correlation_df(data)
df.shape

def convert_to_phase(site):
    return list(map(lambda x: m.atan(x), site))

def calculate_mean(site):
    return (np.mean(site))

def calculate_dev(site):
    return site - calculate_mean(site)

dev_site1 = calculate_dev(convert_to_phase(data[0]))
dev_site2 = calculate_dev(convert_to_phase(data[1]))
dev_site3 = calculate_dev(convert_to_phase(data[2]))
dev_site4 = calculate_dev(convert_to_phase(data[3]))

dev_data = [dev_site1, dev_site2, dev_site3, dev_site4]


#Correlation between original vectors
get_correlation_df(data).corr()

temp_data = [convert_to_phase(data[0]), convert_to_phase(data[1]), convert_to_phase(data[2]), convert_to_phase(data[3])]
get_correlation_df(data).corr()

#Correlation between deviation
get_correlation_df(dev_data).corr()

from scipy.stats.mstats import mquantiles
np.set_printoptions(precision=3)
[mquantiles(x,prob=[0,0.25,0.5,0.75,1]) for x in data]

[np.std(x) for x in data]

[np.min(np.diff(np.sort(np.unique(x)))) for x in data]

data_mad = list(map(swp.mad,data))
data_mad

data = list(map(lambda x: (x-np.median(x))/swp.mad(x), data))


from scipy.signal import fftconvolve
from numpy import apply_along_axis as apply 
data_filtered = apply(lambda x:
                      fftconvolve(x,np.array([1,1,1,1,1])/5.,'same'),
                      1,np.array(data))
data_filtered = (data_filtered.transpose() / \
                 apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0

sp0 = swp.peak(data_filtered.sum(0))

round(np.mean(np.diff(sp0)))
round(np.std(np.diff(sp0)))

np.min(np.diff(sp0))

np.max(np.diff(sp0))


sp0E = sp0[sp0 <= data_len/2.]
len(sp0E)

sp0L = sp0[sp0 > data_len/2.]
len(sp0L)

evtsE = swp.mk_events(sp0E,np.array(data),49,50)
evtsE_median=apply(np.median,0,evtsE)
evtsE_mad=apply(swp.mad,0,evtsE)

evtsE = swp.mk_events(sp0E,np.array(data),14,30)

noiseE = swp.mk_noise(sp0E,np.array(data),14,30,safety_factor=2.5,size=2000)


def good_evts_fct(samp, thr=3):
    samp_med = apply(np.median,0,samp)
    samp_mad = apply(swp.mad,0,samp)
    above = samp_med > 0
    samp_r = samp.copy()
    for i in range(samp.shape[0]): samp_r[i,above] = 0
    samp_med[above] = 0
    res = apply(lambda x:
                np.all(abs((x-samp_med)/samp_mad) < thr),
                1,samp_r)
    return res

goodEvts = good_evts_fct(evtsE,8)

from numpy.linalg import svd
varcovmat = np.cov(evtsE[goodEvts,:].T)
u, s, v = svd(varcovmat)

noiseVar = sum(np.diag(np.cov(noiseE.T)))
evtsVar = sum(s)
[(i,sum(s[:i])+noiseVar-evtsVar) for i in range(15)]

from sklearn.cluster import KMeans
km10 = KMeans(n_clusters=10, init='k-means++', n_init=100, max_iter=100)
km10.fit(np.dot(evtsE[goodEvts,:],u[:,0:3]))
c10 = km10.fit_predict(np.dot(evtsE[goodEvts,:],u[:,0:3]))

cluster_median = list([(i,
                        np.apply_along_axis(np.median,0,
                                            evtsE[goodEvts,:][c10 == i,:]))
                                            for i in range(10)
                                            if sum(c10 == i) > 0])
cluster_size = list([np.sum(np.abs(x[1])) for x in cluster_median])
new_order = list(reversed(np.argsort(cluster_size)))
new_order_reverse = sorted(range(len(new_order)), key=new_order.__getitem__)
c10b = [new_order_reverse[i] for i in c10]

centers = { "Cluster " + str(i) :
            swp.mk_center_dictionary(sp0E[goodEvts][np.array(c10b)==i],
                                     np.array(data))
            for i in range(10)}

swp.classify_and_align_evt(sp0[0],np.array(data),centers)

data0 = np.array(data) 
round0 = [swp.classify_and_align_evt(sp0[i],data0,centers)
          for i in range(len(sp0))]

len([x[1] for x in round0 if x[0] == '?'])

pred0 = swp.predict_data(round0,centers)
pred0

data1 = data0 - pred0
data1

data_filtered = np.apply_along_axis(lambda x:
                                    fftconvolve(x,np.array([1,1,1])/3.,
                                                'same'),
                                    1,data1)
data_filtered = (data_filtered.transpose() /
                 np.apply_along_axis(swp.mad,1,
                                     data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0
sp1 = swp.peak(data_filtered[0,:])

round1 = [swp.classify_and_align_evt(sp1[i],data1,centers)
          for i in range(len(sp1))]
pred1 = swp.predict_data(round1,centers)
data2 = data1 - pred1

len([x[1] for x in round1 if x[0] == '?'])


data_filtered = apply(lambda x:
                      fftconvolve(x,np.array([1,1,1])/3.,'same'),
                      1,data2)
data_filtered = (data_filtered.transpose() / \
                 apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0
sp2 = swp.peak(data_filtered[1,:])
len(sp2)

round2 = [swp.classify_and_align_evt(sp2[i],data2,centers) for i in range(len(sp2))]
pred2 = swp.predict_data(round2,centers)
data3 = data2 - pred2
len([x[1] for x in round2 if x[0] == '?'])

data_filtered = apply(lambda x:
                      fftconvolve(x,np.array([1,1,1])/3.,'same'),
                      1,data3)
data_filtered = (data_filtered.transpose() / \
                 apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0
sp3 = swp.peak(data_filtered[2,:])
len(sp3)


round3 = [swp.classify_and_align_evt(sp3[i],data3,centers) for i in range(len(sp3))]
pred3 = swp.predict_data(round3,centers)
data4 = data3 - pred3
len([x[1] for x in round3 if x[0] == '?'])

data_filtered = apply(lambda x:
                      fftconvolve(x,np.array([1,1,1])/3.,'same'),
                      1,data4)
data_filtered = (data_filtered.transpose() / \
                 apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0
sp4 = swp.peak(data_filtered[3,:])
len(sp4)

round4 = [swp.classify_and_align_evt(sp4[i],data4,centers) for i in range(len(sp4))]
pred4 = swp.predict_data(round4,centers)
data5 = data4 - pred4
len([x[1] for x in round4 if x[0] == '?'])

round_all = round0.copy() # Python 3
# round_all = round0[:] # Python 2
round_all.extend(round1)
round_all.extend(round2)
round_all.extend(round3)
round_all.extend(round4)
#spike_trains = { n : np.sort([x[1] + x[2] for x in round_all
#                              if x[0] == n]) for n in list(centers)}

# Create spike times dictionary first
spike_times = {
    n: np.sort([x[1] + x[2] for x in round_all if x[0] == n])
    for n in list(centers)
}

spike_trains = {
    n: swp.mk_events([int(t) for t in spike_times[n]], data4, 14, 30)
    for n in spike_times
}

[(n,len(spike_trains[n])) for n in list(centers)]



