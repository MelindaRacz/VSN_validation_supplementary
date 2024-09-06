import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import FastICA

ifn = "VEP_VSN_2024-06-06_12-38-51.csv"
#ifn = "VEP_Smarting_2024-08-01_13-40-25.csv"

data_str = np.loadtxt(ifn, dtype=str, delimiter=';')
data = np.zeros(data_str.shape)
fs = 250

if ifn.split(sep='_')[1] == "VSN":
    data = data_str[:, 0:8].astype(np.float32) * 0.045
    data = np.delete(data, obj=6, axis=1)
else:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = float(data_str[i, j].replace(',', '.'))
    data = data[:, 0:-2]

    ref = data[:, 5]
    for i in range(data.shape[1]):
        if i != 5:
            data[:, i] = data[:, i] - ref
    data = np.delete(data, obj=[5, 6], axis=1)
    fs = 500

phase = data_str[:, -2].astype(int)
i_trial = data_str[:, -1].astype(int)

edges = np.insert(i_trial[1:i_trial.shape[0]] - i_trial[0:-1], obj=0, values=0)
edges[edges.shape[0] - 1] = 1
i_edges = np.where(edges == 1)[0]
trial_lengths = i_edges[1:i_edges.shape[0]] - i_edges[0:-1]
bad_trial_threshold = 3
bad_trials = np.where(np.bitwise_or(trial_lengths > trial_lengths.mean() + bad_trial_threshold * trial_lengths.std(), trial_lengths < trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()))[0]
trial_length = np.min(np.delete(trial_lengths, bad_trials))
n_trials = i_edges.shape[0] - 1

hp = signal.butter(2, 0.5, 'hp', fs=fs, output='sos')
lp = signal.butter(2, 25, 'lp', fs=fs, output='sos')

for i in range(data.shape[1]):
    data[:, i] = signal.sosfiltfilt(hp, data[:, i])
    data[:, i] = signal.sosfiltfilt(lp, data[:, i])

random_state = 23
ica = FastICA(n_components=None, random_state=random_state)
S_ = ica.fit_transform(data)
S_plot = np.zeros(S_.shape)

for j in range(S_.shape[1]):
    S_plot[:, j] = S_[:, j] + 50 * j
plt.plot(S_plot)
plt.show()

sources_to_delete = [5]
for j in sources_to_delete:
    S_[:, int(j)] = 0

data = ica.inverse_transform(S_)

#bad_trial_margin = 6
#bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(data > data.mean() + bad_trial_margin * data.std(), data < data.mean() - bad_trial_margin * data.std()))[0]])
threshold = 50
bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(data > data.mean() + threshold, data < data.mean() - threshold))[0]])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(data)
# ax1.plot((data.mean() + bad_trial_margin * data.std()) * np.ones(data.shape[0]))
# ax1.plot((data.mean() - bad_trial_margin * data.std()) * np.ones(data.shape[0]))
ax1.plot((data.mean() + threshold) * np.ones(data.shape[0]))
ax1.plot((data.mean() - threshold) * np.ones(data.shape[0]))
ax2.plot(phase - 2)
ax2.plot(edges)
ax2.plot(i_trial + 2)
plt.show()

print(i_edges)
print(bad_trials)
print(bad_trials2)
print(trial_length)
print(n_trials)
plt.plot(trial_lengths)
plt.plot((trial_lengths.mean() + bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.plot((trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.show()

trial_data = np.zeros((n_trials, trial_length, data.shape[1]))
for i in range(n_trials):
    if not ((i_edges[i] in bad_trials) or (i_edges[i] in bad_trials2)):
        trial_data[i, :, :] = data[i_edges[i]:i_edges[i] + trial_length, :]
target_data = np.delete(trial_data, obj=np.where(np.sum(np.sum(trial_data, axis=1), axis=1) == 0)[0], axis=0)

# for i in range(trial_data.shape[0]):
#     for j in range(trial_data.shape[2]):
#         trial_data[i, :, j] = signal.sosfiltfilt(hp, trial_data[i, :, j])
#         trial_data[i, :, j] = signal.sosfiltfilt(bs50, trial_data[i, :, j])
#         trial_data[i, :, j] = signal.sosfiltfilt(bs100, trial_data[i, :, j])

trial_average = trial_data.mean(axis=0)
for i in range(trial_average.shape[1]):
    trial_average[:, i] = trial_average[:, i] - trial_average[:, i].mean()
for i in range(trial_average.shape[1]):
    trial_average[:, i] = trial_average[:, i] - trial_average.mean(axis=1)

# VEP_start = int(13 * fs / 250)
# VEP_end = int(90 * fs / 250)
VEP_start = 0
VEP_end = int(trial_average.shape[0]/2)

plt.plot(trial_average)
plt.vlines([VEP_start, VEP_end], np.min(trial_average), np.max(trial_average))
plt.show()

VEP_occipital = trial_average[:, 5:trial_average.shape[1]]
for i in range(VEP_occipital.shape[1]):
    VEP_occipital[:, i] = VEP_occipital[:, i] - np.mean(trial_average[:, 0:2], axis=1)
amp_VEP = np.ptp(VEP_occipital[VEP_start:VEP_end, :], axis=0)
amp_noise = np.ptp(VEP_occipital[VEP_end:VEP_occipital.shape[0], :], axis=0)
print(amp_VEP)
print(amp_noise)
print(amp_VEP/amp_noise)

plt.plot(VEP_occipital)
plt.vlines([VEP_start, VEP_end], np.min(VEP_occipital), np.max(VEP_occipital))
plt.hlines([np.min(VEP_occipital[VEP_start:VEP_end, :]),
               np.max(VEP_occipital[VEP_start:VEP_end, :]),
               np.min(VEP_occipital[VEP_end:VEP_occipital.shape[0], :]),
               np.max(VEP_occipital[VEP_end:VEP_occipital.shape[0], :])], 0, VEP_occipital.shape[0])
plt.grid()
plt.show()

fft_signal = VEP_occipital[VEP_start:VEP_end, :]
fft_noise = VEP_occipital[VEP_end:VEP_occipital.shape[0], :]
for i in range(VEP_occipital.shape[1]):
    fft_signal[:, i] = np.square(np.abs(fft(fft_signal[:, i])))
    fft_noise[:, i] = np.square(np.abs(fft(fft_noise[:, i])))

xf = fftfreq(VEP_end, 1/fs)[:VEP_end//2]
fft_signal = fft_signal[0:VEP_end//2]
fft_noise = fft_noise[0:VEP_end//2]

plt.plot(xf, fft_signal)
plt.plot(xf, fft_noise)
plt.show()

SNR = 10 * np.log10(fft_signal/fft_noise)
k_lower = int(np.round(2 * 1 * SNR.shape[0] / fs))
k_upper = int(np.round(2 * 25 * SNR.shape[0] / fs))

plt.plot(SNR)
plt.show()
SNR_sum = np.sum(SNR[k_lower + 1:k_upper + 1, :], axis=0)
print(SNR_sum)
plt.plot(SNR[k_lower + 1:k_upper + 1, :])
plt.show()

