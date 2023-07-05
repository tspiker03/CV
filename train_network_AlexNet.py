import matplotlib
import time
matplotlib.use("Agg")

# import the necessary packages
from AlexNet import AlexNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
dataset = "/Users/susanwhite/PycharmProjects/CNN_Comparisons/dataset"

epochs = 50
tic = time.perf_counter()
print("[INFO] loading images...")
#define the path to the dataset and load it
imagePaths = list(paths.list_images('/Users/susanwhite/PycharmProjects/CNN_Comparisons/dataset'))
data = []
labels = []


# loop over the image paths, convert images to grayscale, resize to 64x64
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0


#convert classes to integers, then to vectors
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# train test split
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.40, stratify=labels, random_state=0)

# Adam optimizer
print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-4, weight_decay=1e-4 / epochs)
model = AlexNet.build(width=64, height=64, depth=1,
                    classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(epochs))
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
              batch_size=32, epochs=epochs, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
toc = time.perf_counter()
print(f" ran the CNN in {toc - tic:0.4f} seconds")

print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("AlexNet_plot")
f = open("AlexNet.txt", "a")
f.write(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
f.write(f" ran the CNN in {toc - tic:0.4f} seconds")
f.close()


idxs = np.arange(0, testY.shape[0])
idxs = np.random.choice(idxs, size=(25,), replace=False)
images = []

# loop over the testing indexes
for i in idxs:
    # grab the current testing image and classify it
    image = np.expand_dims(testX[i], axis=0)
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = le.classes_[j]

    # rescale the image into the range [0, 255] and then resize it so
    # we can more easily visualize it
    output = (image[0] * 255).astype("uint8")
    output = np.dstack([output] * 3)
    output = cv2.resize(output, (128, 128))

    # draw the colored class label on the output image and add it to
    # the set of output images
    color = (0, 0, 255) if "non" in label else (0, 255, 0)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    images.append(output)

# create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]

# show the output montage
cv2.imshow("Output", montage)
#cv2.imwrite("AlexNet_output", montage)
cv2.waitKey(0)
