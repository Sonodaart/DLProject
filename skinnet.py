#################################################################################
# This file implements base functions for:
# 	- loading the datasets
# 	- displaying some samples of the datasets
# 	- building a model, given some hyperparameters in input
# 	- evaluating model perfomances on the two tasks
# 	- training the model
#################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, RandomFlip, RandomRotation


# ============================== Loading Dataset ==============================
def load_data(BATCH_SIZE):
	COLAB = True
	DATASET_FIRST = True

	if DATASET_FIRST:
		name = "first"
	else:
		name = "second"
	if COLAB:
		train_dir = f"./data_{name}/training/"
		validation_dir = f"./data_{name}/validation/"
		t1test_dir = f"./data_{name}/task1/testing"
		t2test_dir = f"./data_{name}/task2/testing"
	else:
		train_dir = f"/kaggle/input/data{name}/data/training/"
		validation_dir = f"/kaggle/input/data{name}/data/validation/"
		t1test_dir = f"/kaggle/input/data{name}/data/task1/testing"
		t2test_dir = f"/kaggle/input/data{name}/data/task2/testing"
	IMG_SIZE = (299,299)
	NUM_CLASSES = len([x[1] for x in os.walk(train_dir)][0])

	train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
																	shuffle=True,
																	batch_size=BATCH_SIZE,
																	image_size=IMG_SIZE)

	validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
																		shuffle=True,
																		batch_size=BATCH_SIZE,
																		image_size=IMG_SIZE)

	task1_test_dataset = tf.keras.utils.image_dataset_from_directory(t1test_dir, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

	task2_test_dataset = tf.keras.utils.image_dataset_from_directory(t2test_dir, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

	class_names = train_dataset.class_names
	# the order of the two classes is important
	assert(task1_test_dataset.class_names==["kerato","sebo"])
	assert(task2_test_dataset.class_names==["melanoma","nevi"])


	# Classes to merge in order to answer the two tasks
	kerato = ["Basal cell carcinoma","Malignant epidermal proliferations","Keratoacanthoma",
				"Squamous cell carcinoma in situ","Squamous cell carcinoma, Invasive","Squamous cell carcinoma, NOS","Kerato"]

	sebo = ["Seborrheic keratosis","Pigmented benign keratosis","Benign epidermal proliferations"]

	melanoma = ["Melanoma, NOS", "Melanoma metastasis","Melanoma"]

	nevi = ["Nevus","Benign melanocytic proliferations"]

	# indices of the class_names vector, of the relative supclasses
	t1kerato_index = np.array([i for i in range(len(class_names)) if class_names[i] in kerato])
	t1sebo_index = np.array([i for i in range(len(class_names)) if class_names[i] in sebo])
	t2melanoma_index = np.array([i for i in range(len(class_names)) if class_names[i] in melanoma])
	t2nevi_index = np.array([i for i in range(len(class_names)) if class_names[i] in nevi])

	train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

	task1_test_dataset = task1_test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

	task2_test_dataset = task2_test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

	return class_names, t1kerato_index, t1sebo_index, t2melanoma_index, t2nevi_index, \
			train_dataset, validation_dataset, \
			task1_test_dataset, task2_test_dataset


def display_images(dataset, class_names):
	# display of some imgaes
	plt.figure(figsize=(10, 10))
	for images, labels in dataset.take(1):
		for i in range(25):
			ax = plt.subplot(5, 5, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(class_names[labels[i]])
			plt.axis("off")
	plt.show()

	# display of a preprocessed image various times
	data_augmentation = tf.keras.Sequential([
		RandomFlip('vertical'),
		RandomRotation(1.),
	])

	for image, _ in dataset.take(1):
	  plt.figure(figsize=(10, 10))
	  first_image = image[0]
	  for i in range(25):
	    ax = plt.subplot(5, 5, i + 1)
	    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
	    plt.imshow(augmented_image[0] / 255)
	    plt.axis('off')
	plt.show()


# ============================== Building Model ==============================
def skinnet(NUM_CLASSES, params, input_shape=(299,299,3), verbose=False):
	inceptionv3_model = InceptionV3(weights='imagenet', include_top=False)
	inceptionv3_model.trainable = False
	if verbose:
		print(base_model.summary())

	data_augmentation = tf.keras.Sequential([
		RandomFlip('vertical'),
		RandomRotation(1.),
	])

	# Model construction
	inputs = Input(shape=input_shape)
	x = data_augmentation(inputs)
	x = tf.keras.applications.inception_v3.preprocess_input(x)
	x = inceptionv3_model(x, training=False)
	x = GlobalAveragePooling2D()(x)
	# first layer
	x = Dropout(rate=0.2)(x)
	x = Dense(units=int(params["layer_depth"]["units1"]), activation='relu')(x)
	# second layer
	if params["layer_depth"]["type"] in ["two_layers","three_layers"]:
		x = Dropout(rate=0.2)(x)
		x = Dense(units=int(params["layer_depth"]["units2"]), activation='relu')(x)
	# third layer
	if params["layer_depth"]["type"]=="three_layers":
		x = Dropout(rate=0.2)(x)
		x = Dense(units=int(params["layer_depth"]["units3"]), activation='relu')(x)
	predictions = Dense(NUM_CLASSES, activation='softmax')(x)

	model = Model(inputs=inputs, outputs=predictions)

	# optimizer parameters
	if params["opt_config"]["type"]=="RMSprop":
		model.compile(
			optimizer=tf.keras.optimizers.RMSprop(learning_rate=params["opt_config"]["lr"],
													weight_decay=params["opt_config"]["decay"], momentum=params["opt_config"]["momentum"],
													epsilon=params["opt_config"]["epsilon"]),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			metrics=['accuracy'])
	if params["opt_config"]["type"]=="Adam":
		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=params["opt_config"]["lr"]),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
			metrics=['accuracy'])

	if verbose:
		print(model.summary())
	return model


# ============================== Evaluating ==============================
def test(model, t1kerato_index, t1sebo_index, t2melanoma_index, t2nevi_index,
			task1_test_dataset, task2_test_dataset):
	# Task 1
	acc_t1_test = 0
	n = 0
	for x,y in task1_test_dataset:
		print(y.shape)
		pred = model(x,training=False).numpy()
		kerato_pred = np.sum(pred[:,t1kerato_index],axis=1)	
		sebo_pred = np.sum(pred[:,t1sebo_index],axis=1)
		# I asserted for 0=kerato, 1=sebo
		pred = (kerato_pred<sebo_pred).astype(np.int32)
		acc_t1_test += np.sum(pred==y)
		n += y.shape[0]
	acc_t1_test = acc_t1_test/n

	# Task 2
	acc_t2_test = 0
	n = 0
	for x,y in task2_test_dataset:
		pred = model(x,training=False).numpy()
		melanoma_pred = np.sum(pred[:,t2melanoma_index],axis=1)	
		nevi_pred = np.sum(pred[:,t2nevi_index],axis=1)
		# I asserted for 0=melanoma, 1=nevi
		pred = (melanoma_pred<nevi_pred).astype(np.int32)
		acc_t2_test += np.sum(pred==y)
		n += y.shape[0]
	acc_t2_test = acc_t2_test/n

	return [acc_t1_test,acc_t2_test]


# ============================== Training Model ==============================
def train(model, train_dataset, validation_dataset, epochs=50, save_weights=True, patience=5, verbose="auto"):
	history = model.fit(train_dataset,
						epochs=epochs,
						verbose=verbose,
						validation_data=validation_dataset,
						callbacks=[
						tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
							tf.keras.callbacks.TensorBoard(log_dir='./tblog'),
						],)
	if save_weights:
		model.save('my_model.keras')

	if verbose==0:
		_ = len(history.history["val_accuracy"])
		print(f"finished in {_} epochs")
	return history.history["val_accuracy"][-1], history.history["val_loss"][-1]



