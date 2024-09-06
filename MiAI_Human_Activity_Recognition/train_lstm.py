import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

topspin_df = pd.read_csv("TOPSPIN.txt")
topspinbh_df = pd.read_csv("TOPSPINBACKHAND.txt")
push_df = pd.read_csv("PUSH.txt")
pushbh_df = pd.read_csv("PUSHBACKHAND.txt")
# blockbh_df = pd.read_csv("BLOCKBACKHAND.txt")

X = []
y = []
no_of_timesteps = 4

dataset = topspin_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0,0])

dataset = topspinbh_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0])

dataset = push_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,1,0])

dataset = pushbh_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,1])

# dataset = blockbh_df.iloc[:,1:].values
# n_sample = len(dataset)
# for i in range(no_of_timesteps, n_sample):
#     X.append(dataset[i-no_of_timesteps:i,:])
#     y.append([0,0,0,1])

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 4, activation="softmax"))
# model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")
# model.compile(optimizer="adam", metrics = ['accuracy'], loss = "sparse_categorical_crossentropy")

history = model.fit(X_train, y_train, epochs=1000, batch_size=4196, validation_data=(X_test, y_test),
# model.fit(X_train, y_train, epochs=1500, batch_size=368, validation_data=(X_test, y_test),
          callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10, restore_best_weights=True)],)
model.save("model.h5")

# Training and Validation Loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Training and Validation Accuracy
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


