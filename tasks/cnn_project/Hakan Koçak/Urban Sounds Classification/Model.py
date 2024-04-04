import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


images = []
labels_str = []

def listeleme(kok_dizin):
    for sinif in os.listdir(kok_dizin):
        print(sinif)
        sinif_dizini = os.path.join(kok_dizin, sinif)
        if os.path.isdir(sinif_dizini):
            for dosya in os.listdir(sinif_dizini):
                if dosya.endswith(".png"):
                    yol = os.path.join(sinif_dizini, dosya)
                    img = cv2.imread(yol)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img / 255
                    images.append(img)
                    labels_str.append(sinif)

kok_dizin = "Processed_Spectrograms"
listeleme(kok_dizin)



numaralandirma = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9
}

labels = [numaralandirma.get(oges, None) for oges in labels_str]

images = np.array(images)
labels = np.array(labels)

X_train, X_, y_train, y_ = train_test_split(images, labels, train_size=0.7, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_, y_, test_size=0.5, random_state=42)


#MODEL
model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                    input_shape=[100, 100, 1]))
model_1.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model_1.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(128, activation='relu'))
model_1.add(tf.keras.layers.Dense(64, activation='relu'))
model_1.add(tf.keras.layers.Dense(10, activation='softmax'))
model_1.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model_2 = tf.keras.models.Sequential()
model_2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                    input_shape=[100, 100, 1]))
model_2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_2.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model_2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(128, activation='relu'))
model_2.add(tf.keras.layers.Dense(64, activation='relu'))
model_2.add(tf.keras.layers.Dense(10, activation='softmax'))
model_2.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_3 = tf.keras.models.Sequential()
model_3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                    input_shape=[100, 100, 1]))
model_3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model_3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_3.add(tf.keras.layers.Flatten())
model_3.add(tf.keras.layers.Dense(128, activation='relu'))
model_3.add(tf.keras.layers.Dense(10, activation='softmax'))
model_3.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_4 = tf.keras.models.Sequential()
model_4.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                 input_shape=[100, 100, 1]))
model_4.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_4.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model_4.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model_4.add(tf.keras.layers.Flatten())
model_4.add(tf.keras.layers.Dense(128, activation='relu'))
model_4.add(tf.keras.layers.Dense(10, activation='softmax'))
model_4.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])




accuracy_score_train = []
loss_train = []
accuracy_score_val = []
loss_val = []


models = [model_1, model_2, model_3, model_4]


for model in models:
    tmp = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    accuracy_score_train.append(tmp.history['accuracy'])
    loss_train.append(tmp.history['loss'])
    accuracy_score_val.append(tmp.history['val_accuracy'])
    loss_val.append(tmp.history['val_loss'])


accuracy_score_train = np.array(accuracy_score_train)
loss_train = np.array(loss_train)
accuracy_score_val = np.array(accuracy_score_val)
loss_val = np.array(loss_val)



plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))
for i, ax in enumerate(axes.flat):
    if i % 2 == 0:
        ax.plot(accuracy_score_train[i // 2])
        ax.plot(accuracy_score_val[i // 2])
        ax.set_title(f'Model {i // 2 + 1} Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(10))
        ax.set_xticklabels(np.arange(1, 11))
        ax.legend(['Train', 'Validation'])
    else:
        ax.plot(loss_train[i // 2])
        ax.plot(loss_val[i // 2])
        ax.set_title(f'Model {i // 2 + 1} Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_xticks(range(10))
        ax.set_xticklabels(np.arange(1, 11))
        ax.legend(['Train', 'Validation'])
img = plt.tight_layout()
plt.savefig('metric_graph.png')
plt.show()



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                                input_shape=[100, 100, 1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=7, validation_data=(X_val, y_val))

model.save('cnn_model.h5')  # Sürekli Sürekli Eğitmemek için kaydettik.
"""
loaded_model = load_model('cnn_model.h5')
"""

predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print(f"Accuracy: {round(accuracy_score(y_test, predicted_classes), 2)}")
print(f"Recall: {round(recall_score(y_test,predicted_classes, average='weighted'),3)}")
print(f"Precision: {round(precision_score(y_test,predicted_classes, average='weighted'), 2)}")
print(f"F1: {round(f1_score(y_test,predicted_classes, average='weighted'), 2)}")




#Accuracy: 0.87
#Recall: 0.866
#Precision: 0.87
#F1: 0.87
