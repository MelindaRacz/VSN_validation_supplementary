import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

ifn = "P300_VSN_2024-06-20_13-48-52.csv"
#ifn = "P300_Smarting_2024-08-01_13-49-01.csv"

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
    data = data[:, 0:-3]

    ref = data[:, 5]
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] - ref
    data = np.delete(data, obj=[5, 6], axis=1)

    fs = 500

#key_pressed = data_str[:, -3].astype(int)
phase = data_str[:, -2].astype(int)
i_trial = data_str[:, -1].astype(int)
edges = np.insert(i_trial[1:i_trial.shape[0]] - i_trial[0:-1], obj=0, values=0)
edges[edges.shape[0] - 1] = 1

hp = signal.butter(2, 1, 'hp', fs=fs, output='sos')
lp = signal.butter(2, 15, 'lp', fs=fs, output='sos')

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

sources_to_delete = [3]
for j in sources_to_delete:
    S_[:, int(j)] = 0

data = ica.inverse_transform(S_)

# bad_trial_margin = 7
# bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(data > data.mean() + bad_trial_margin * data.std(), data < data.mean() - bad_trial_margin * data.std()))[0]])
threshold = 50
bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(data > data.mean() + threshold, data < data.mean() - threshold))[0]])

plt.plot(data)
plt.plot(phase * np.ptp(data))
plt.plot(edges * np.ptp(data))
# plt.plot((data.mean() + bad_trial_margin * data.std()) * np.ones(data.shape[0]))
# plt.plot((data.mean() - bad_trial_margin * data.std()) * np.ones(data.shape[0]))
plt.plot((data.mean() + threshold) * np.ones(data.shape[0]))
plt.plot((data.mean() - threshold) * np.ones(data.shape[0]))
plt.show()

i_edges = np.where(edges == 1)[0]
trial_lengths = i_edges[1:i_edges.shape[0]] - i_edges[0:-1]
bad_trial_threshold = 3
bad_trials = np.where(np.bitwise_or(trial_lengths > trial_lengths.mean() + bad_trial_threshold * trial_lengths.std(), trial_lengths < trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()))[0]
trial_length = np.min(np.delete(trial_lengths, bad_trials))
n_trials = i_edges.shape[0] - 1
trial_labels = phase[i_edges[0:i_edges.shape[0]-1] + int(np.round(trial_length/2))]
i_targets = np.where(trial_labels == 1)[0]
i_other = np.where(trial_labels == 0)[0]

print(i_edges)
print(trial_length)
print(bad_trials)
print(bad_trials2)
print(n_trials)
print(trial_labels)
print(i_targets)
print(i_other)

plt.plot(trial_lengths)
plt.plot((trial_lengths.mean() + bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.plot((trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.show()

target_data = np.zeros((i_targets.shape[0], trial_length, data.shape[1]))
for i in range(i_targets.shape[0]):
    if not ((i_targets[i] in bad_trials) or (i_targets[i] in bad_trials2)):
        target_data[i, :, :] = data[i_edges[i_targets[i]]:i_edges[i_targets[i]] + trial_length, :]
target_data = np.delete(target_data, obj=np.where(np.sum(np.sum(target_data, axis=1), axis=1) == 0)[0], axis=0)

other_data = np.zeros((i_other.shape[0], trial_length, data.shape[1]))
for i in range(i_other.shape[0]):
    if not ((i_other[i] in bad_trials) or (i_other[i] in bad_trials2)):
        other_data[i, :, :] = data[i_edges[i_other[i]]:i_edges[i_other[i]] + trial_length, :]
other_data = np.delete(other_data, obj=np.where(np.sum(np.sum(other_data, axis=1), axis=1) == 0,)[0], axis=0)

target_average = target_data.mean(axis=0)
other_average = other_data.mean(axis=0)
# for i in range(target_average.shape[1]):
#     target_average[:, i] = target_average[:, i] - target_average[:, i][0]
# for i in range(other_average.shape[1]):
#     other_average[:, i] = other_average[:, i] - other_average[:, i][0]

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(target_average[0:fs, :])
# ax2.plot(other_average[0:fs, :])
# #ax3.plot((target_average - other_average)[0:250, 2:5])
# ax3.plot(target_average[0:fs, :], linewidth=1.5)
# ax3.plot(other_average[0:fs, :], linewidth=0.5)
# plt.show()

P300_start = 80
P300_end = 150
# P300_start = 228
# P300_end = 325

plt.plot(target_average[0:fs, :], linewidth=1.5)
plt.plot(other_average[0:fs, :], linewidth=0.5)
plt.vlines([P300_start, P300_end], np.min((target_average, other_average)), np.max((target_average, other_average)))
plt.show()

target = target_average[0:fs, 2:5]
other = other_average[0:fs, 2:5]

target = target_average[P300_start:P300_end, 2:5]
other = other_average[P300_start:P300_end, 2:5]
amp_target = np.ptp(target, axis=0)
amp_other = np.ptp(other, axis=0)

print(amp_target)
print(amp_other)
print(amp_target / amp_other)

plt.plot(target_average[0:fs, 2:5], linewidth=1.5)
plt.plot(other_average[0:fs, 2:5], linewidth=0.5)
plt.vlines([P300_start, P300_end], np.min((target_average, other_average)), np.max((target_average, other_average)))
plt.hlines([np.min(target),
               np.max(target),
               np.min(other),
               np.max(other)], 0, target_average.shape[0])
plt.grid()
plt.show()
