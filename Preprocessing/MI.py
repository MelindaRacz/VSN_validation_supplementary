import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
from scipy.fft import fft
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
from keras import layers

#ifn = "MI_VSN_2024-08-01_13-09-52.csv"
ifn = "MI_Smarting_2024-07-31_14-28-05.csv"

fs = 250
data_str = np.loadtxt(ifn, dtype=str, delimiter=';')
data = np.zeros(data_str.shape)
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
        data[:, i] = data[:, i] - ref
    data = np.delete(data, obj=[5, 6], axis=1)

    fs = 500

phase = data_str[:, -2].astype(int)
i_trial = data_str[:, -1].astype(int)
edges = np.insert(i_trial[1:i_trial.shape[0]] - i_trial[0:-1], obj=0, values=0)
edges[edges.shape[0] - 1] = 1

hp = signal.butter(2, 1, 'hp', fs=fs, output='sos')
lp = signal.butter(2, 40, 'lp', fs=fs, output='sos')
filtered_data = np.zeros(data.shape)
for i in range(data.shape[1]):
    filtered_data[:, i] = data[:, i]

for i in range(data.shape[1]):
    filtered_data[:, i] = signal.sosfiltfilt(hp, filtered_data[:, i])
    filtered_data[:, i] = signal.sosfiltfilt(lp, filtered_data[:, i])

random_state = 23
ica = FastICA(n_components=None, random_state=random_state)
S_ = ica.fit_transform(filtered_data)
S_plot = np.zeros(S_.shape)

for j in range(S_.shape[1]):
    S_plot[:, j] = S_[:, j] + 50 * j
plt.plot(S_plot)
plt.show()

sources_to_delete = [0, 1, 5]
for j in sources_to_delete:
    S_[:, int(j)] = 0

filtered_data = ica.inverse_transform(S_)

# bad_trial_margin = 20
# bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(filtered_data > filtered_data.mean() + bad_trial_margin * filtered_data.std(), filtered_data < filtered_data.mean() - bad_trial_margin * filtered_data.std()))[0]])
threshold = 50
bad_trials2 = np.unique(i_trial[np.where(np.bitwise_or(filtered_data > filtered_data.mean() + threshold, filtered_data < filtered_data.mean() - threshold))[0]])

plt.plot(filtered_data)
plt.plot(phase * np.ptp(filtered_data))
plt.plot(edges * np.ptp(filtered_data))
# plt.plot((filtered_data.mean() + bad_trial_margin * filtered_data.std()) * np.ones(filtered_data.shape[0]))
# plt.plot((filtered_data.mean() - bad_trial_margin * filtered_data.std()) * np.ones(filtered_data.shape[0]))
plt.plot((filtered_data.mean() + threshold) * np.ones(data.shape[0]))
plt.plot((filtered_data.mean() - threshold) * np.ones(data.shape[0]))
plt.show()
plt.show()

i_edges = np.where(edges == 1)[0]
trial_lengths = i_edges[1:i_edges.shape[0]] - i_edges[0:-1]
bad_trial_threshold = 2
bad_trials = np.where(np.bitwise_or(trial_lengths > trial_lengths.mean() + bad_trial_threshold * trial_lengths.std(), trial_lengths < trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()))[0]
trial_length = np.min(np.delete(trial_lengths, bad_trials))
n_trials = i_edges.shape[0] - 1
trial_labels = phase[i_edges[0:i_edges.shape[0]-1] + int(np.round(trial_length/2))]
i_rest = np.where(trial_labels == 0)[0]
i_left = np.where(trial_labels == 1)[0]
i_right = np.where(trial_labels == 2)[0]

print(i_edges)
print(trial_length)
print(bad_trials)
print(bad_trials2)
print(n_trials)
print(trial_labels)
print(i_rest)
print(i_left)
print(i_right)

plt.plot(trial_lengths)
plt.plot((trial_lengths.mean() + bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.plot((trial_lengths.mean() - bad_trial_threshold * trial_lengths.std()) * np.ones(trial_lengths.shape))
plt.show()

rest_data = np.zeros((i_rest.shape[0], trial_length, data.shape[1]))
for i in range(i_rest.shape[0]):
    if not ((i_rest[i] in bad_trials) or (i_rest[i] in bad_trials2)):
        rest_data[i, :, :] = filtered_data[i_edges[i_rest[i]]:i_edges[i_rest[i]] + trial_length, :]
rest_data = np.delete(rest_data, obj=np.where(np.sum(np.sum(rest_data, axis=1), axis=1) == 0)[0], axis=0)

left_data = np.zeros((i_left.shape[0], trial_length, data.shape[1]))
for i in range(i_left.shape[0]):
    if not ((i_left[i] in bad_trials) or (i_left[i] in bad_trials2)):
        left_data[i, :, :] = filtered_data[i_edges[i_left[i]]:i_edges[i_left[i]] + trial_length, :]
left_data = np.delete(left_data, obj=np.where(np.sum(np.sum(left_data, axis=1), axis=1) == 0)[0], axis=0)

right_data = np.zeros((i_right.shape[0], trial_length, data.shape[1]))
for i in range(i_right.shape[0]):
    if not ((i_right[i] in bad_trials) or (i_right[i] in bad_trials2)):
        right_data[i, :, :] = filtered_data[i_edges[i_right[i]]:i_edges[i_right[i]] + trial_length, :]
right_data = np.delete(right_data, obj=np.where(np.sum(np.sum(right_data, axis=1), axis=1) == 0)[0], axis=0)

# rest_average = rest_data.mean(axis=0)
# left_average = left_data.mean(axis=0)
# right_average = right_data.mean(axis=0)
#
# for i in range(data.shape[1]):
#     rest_average[:, i] = rest_average[:, i] - rest_average[0, i]
#     left_average[:, i] = left_average[:, i] - left_average[0, i]
#     right_average[:, i] = right_average[:, i] - right_average[0, i]

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#ax1.plot(rest_average)
#ax2.plot(left_average)
#ax3.plot(right_average)
#plt.show()

k_lower = int(np.round(8 * trial_length / fs))
#k_upper = int(np.round(int(np.floor(fs/2)) * trial_length / fs))
k_upper = int(np.round(30 * trial_length / fs))

print(k_lower)
print(k_upper)

for i in range(rest_data.shape[0]):
    for j in range(rest_data.shape[2]):
        rest_data[i, :, j] = np.square(np.abs(fft(rest_data[i, :, j])))
rest_data = rest_data[:, k_lower:k_upper, :]
# for i in range(rest_data.shape[0]):
#     rest_data[i, :, :] = rest_data[i, :, :] / np.sum(rest_data[i, :, :])

for i in range(left_data.shape[0]):
    for j in range(left_data.shape[2]):
        left_data[i, :, j] = np.square(np.abs(fft(left_data[i, :, j])))
left_data = left_data[:, k_lower:k_upper, :]
# for i in range(left_data.shape[0]):
#     left_data[i, :, :] = left_data[i, :, :] / np.sum(left_data[i, :, :])

for i in range(right_data.shape[0]):
    for j in range(right_data.shape[2]):
        right_data[i, :, j] = np.square(np.abs(fft(right_data[i, :, j])))
right_data = right_data[:, k_lower:k_upper, :]
# for i in range(right_data.shape[0]):
#     right_data[i, :, :] = right_data[i, :, :] / np.sum(right_data[i, :, :])

rest_data = rest_data[:, :, 2:5]
left_data = left_data[:, :, 2:5]
right_data = right_data[:, :, 2:5]

# for i in range(rest_data.shape[1]):
#     for k in range(rest_data.shape[2]):
#         avg = (np.average(rest_data, axis=0) + np.average(left_data, axis=0) + np.average(left_data, axis=0)) / 3
#         for j in range(rest_data.shape[0]):
#             rest_data[j, i, k] = rest_data[j, i, k] / avg[i, k]
#         for j in range(left_data.shape[0]):
#             left_data[j, i, k] = left_data[j, i, k] / avg[i, k]
#         for j in range(right_data.shape[0]):
#             right_data[j, i, k] = right_data[j, i, k] / avg[i, k]


# for i in range(rest_data.shape[2]):
#     rest_data[:, :, i] = (rest_data[:, :, i] - np.min(rest_data[:, :, i])) / np.std(rest_data[:, :, i])
#     left_data[:, :, i] = (left_data[:, :, i] - np.min(left_data[:, :, i])) / np.std(left_data[:, :, i])
#     right_data[:, :, i] = (right_data[:, :, i] - np.min(right_data[:, :, i])) / np.std(right_data[:, :, i])

for i in range(rest_data.shape[2]):
    rest_data[:, :, i] = (rest_data[:, :, i] - np.min(rest_data[:, :, i])) / (np.max(rest_data[:, :, i]) - np.min(rest_data[:, :, i]))
    left_data[:, :, i] = (left_data[:, :, i] - np.min(left_data[:, :, i])) / (np.max(left_data[:, :, i]) - np.min(left_data[:, :, i]))
    right_data[:, :, i] = (right_data[:, :, i] - np.min(right_data[:, :, i])) / (np.max(right_data[:, :, i]) - np.min(right_data[:, :, i]))

rest_average = rest_data.mean(axis=0)
left_average = left_data.mean(axis=0)
right_average = right_data.mean(axis=0)

print(rest_data.shape)
print(left_data.shape)
print(right_data.shape)

shuffling_vector_rest = np.arange((rest_data.shape[0]))
shuffling_vector_left = np.arange((left_data.shape[0]))
shuffling_vector_right = np.arange((right_data.shape[0]))
np.random.shuffle(shuffling_vector_rest)
np.random.shuffle(shuffling_vector_left)
np.random.shuffle(shuffling_vector_right)

rest_data = rest_data[shuffling_vector_rest, :, :]
left_data = left_data[shuffling_vector_left, :, :]
right_data = right_data[shuffling_vector_right, :, :]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.plot(rest_average)
ax2.plot(left_average)
ax3.plot(right_average)
plt.show()

training_split = 0.8
training_data = np.concatenate((rest_data[0:int(np.round(training_split * rest_data.shape[0])), :, :],
                                      left_data[0:int(np.round(training_split * left_data.shape[0])), :, :],
                                      right_data[0:int(np.round(training_split * right_data.shape[0])), :, :]), axis=0)
test_data = np.concatenate((rest_data[int(np.round(training_split * rest_data.shape[0])):rest_data.shape[0], :, :],
                                   left_data[int(np.round(training_split * left_data.shape[0])):left_data.shape[0], :, :],
                                   right_data[int(np.round(training_split * right_data.shape[0])):right_data.shape[0], :, :]), axis=0)

avg = np.average(np.concatenate((training_data, test_data), axis=0))
std = np.std(np.concatenate((training_data, test_data), axis=0))
training_data = (training_data - avg) / std
test_data = (test_data - avg) / std

training_data2 = np.reshape(training_data, (training_data.shape[0], training_data.shape[1] * training_data.shape[2]))
test_data2 = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
training_labels = np.zeros((training_data.shape[0]), dtype=int)
training_labels[int(np.round(training_split * rest_data.shape[0])):int(np.round(training_split * rest_data.shape[0])) + int(np.round(training_split * left_data.shape[0]))] = 1
training_labels[int(np.round(training_split * rest_data.shape[0])) + int(np.round(training_split * left_data.shape[0])):training_labels.shape[0]] = 2
test_labels = np.zeros((test_data.shape[0]), dtype=int)
test_labels[int(np.round((1 - training_split) * rest_data.shape[0])):int(np.round((1 - training_split) * rest_data.shape[0])) + int(np.round((1 - training_split) * left_data.shape[0]))] = 1
test_labels[int(np.round((1 - training_split) * rest_data.shape[0])) + int(np.round((1 - training_split) * left_data.shape[0])):test_labels.shape[0]] = 2

clf = svm.NuSVC(nu=0.1, kernel='rbf', class_weight='balanced', decision_function_shape='ovo', tol=1e-3, verbose=True)
clf.fit(training_data2, training_labels)
test_labels_predicted = clf.predict(test_data2)
cm = confusion_matrix(test_labels, test_labels_predicted)
print(cm)
print(accuracy_score(test_labels, test_labels_predicted))

clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf.fit(training_data2, training_labels)
test_labels_predicted = clf.predict(test_data2)
cm = confusion_matrix(test_labels, test_labels_predicted)
print(cm)
print(accuracy_score(test_labels, test_labels_predicted))

training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], training_data.shape[2], 1))
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
training_labels_CNN = keras.utils.to_categorical(training_labels, 3)
test_labels_CNN = keras.utils.to_categorical(test_labels, 3)

model = keras.Sequential(
    [
        keras.Input(shape=training_data[0].shape),
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Dropout(0.5),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax"),
    ]
)

batch_size = training_data.shape[0]
epochs = 2500

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1000, start_from_epoch=25, restore_best_weights=True)
train_history = model.fit(training_data, training_labels_CNN, batch_size=batch_size, epochs=epochs, callbacks=callback, validation_split=0.25)
test_labels_predicted = np.argmax(model.predict(test_data), axis=1)
cm = confusion_matrix(test_labels, test_labels_predicted)
print(cm)
score = model.evaluate(test_data, test_labels_CNN, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

loss = train_history.history['loss']
accuracy = train_history.history['accuracy']
val_loss = train_history.history['val_loss']
val_accuracy = train_history.history['val_accuracy']
plt.plot(loss)
plt.plot(accuracy)
plt.plot(val_loss)
plt.plot(val_accuracy)
plt.legend(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
plt.show()
