#################################################################################
# This file hyperoptimize the parameters that the paper didn't specify,
# namely the number of layers of the fully connected in the end, and the
# width of each of those. Additionally, going beyond the paper, even the
# optimizer choice is hyperoptimized. The code, depending on HYPEROPTIMIZE
# either hyperoptimize creating a "trials_all.pkl" file at each step,
# or read the "trials_all.pkl" in the current directory and analyzes it.
#################################################################################

from skinnet import *
from hyperoptimizingTools import *
from hyperopt import hp
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from pickle import load


HYPEROPTIMIZE = False
LOAD = True


tf.keras.utils.set_random_seed(0)

if HYPEROPTIMIZE:
	BATCH_SIZE = 64
	NUM_CLASSES = -1


	data = load_data(BATCH_SIZE)
	class_names = data[0]
	NUM_CLASSES = len(class_names)
	# indices
	t1kerato_index = data[1]
	t1sebo_index = data[2]
	t2melanoma_index = data[3]
	t2nevi_index = data[4]
	# datasets
	train_dataset = data[5]
	validation_dataset = data[6]
	task1_test_dataset = data[7]
	task2_test_dataset = data[8]


# parameter space of the nn of the paper
space = {
	"layer_depth": hp.choice("layer_depth", [
		# --- Configuration for 1 Hidden Layer ---
		{
			"type": "one_layer",
			"units1": hp.quniform("units1_1L", 64, 2048, 32) # Layer 1 size
		},

		# --- Configuration for 2 Hidden Layers ---
		{
			"type": "two_layers",
			"units1": hp.quniform("units1_2L", 64, 2048, 32), # Layer 1 size
			"units2": hp.quniform("units2_2L", 32, 512, 32)  # Layer 2 size
		},

		# --- Configuration for 3 Hidden Layers ---
		{
			"type": "three_layers",
			"units1": hp.quniform("units1_3L", 64, 2048, 32), # Layer 1 size
			"units2": hp.quniform("units2_3L", 32, 512, 32), # Layer 2 size
			"units3": hp.quniform("units3_3L", 16, 128, 16)  # Layer 3 size
		}
		]),
	"opt_config": hp.choice("opt_config", [
		{
			"type": "RMSprop",
			"lr": 0.001,
			"decay": 0.9,
			"momentum": 0.9,
			"epsilon": 0.1
		}])
}

# generic parameter space
space = {
	"layer_depth": hp.choice("layer_depth", [
		# --- Configuration for 1 Hidden Layer ---
		{
			"type": "one_layer",
			"units1": hp.quniform("units1_1L", 64, 2048, 32) # Layer 1 size
		},

		# --- Configuration for 2 Hidden Layers ---
		{
			"type": "two_layers",
			"units1": hp.quniform("units1_2L", 64, 2048, 32), # Layer 1 size
			"units2": hp.quniform("units2_2L", 32, 512, 32)  # Layer 2 size
		},

		# --- Configuration for 3 Hidden Layers ---
		{
			"type": "three_layers",
			"units1": hp.quniform("units1_3L", 64, 2048, 32), # Layer 1 size
			"units2": hp.quniform("units2_3L", 32, 512, 32), # Layer 2 size
			"units3": hp.quniform("units3_3L", 16, 128, 16)  # Layer 3 size
		}
		]),
	"opt_config": hp.choice("opt_config", [
		{
			"type": "Adam",
			"lr": hp.loguniform("learning_rate", -10, 0)
		},
		{
			"type": "RMSprop",
			"lr": hp.loguniform("lr", -10, 0),
			"decay": hp.loguniform("decay", -10, 0),
			"momentum": hp.uniform("momentum", 0, 1),
			"epsilon": hp.loguniform("epsilon", -10, 0)
		}])
}


if LOAD:
	with open("trials_all.pkl","rb") as f:
		trials = load(f)

if HYPEROPTIMIZE:
	runs = 1000
	trials = hyperoptimize_all(space, runs, NUM_CLASSES, train_dataset, validation_dataset,trials_preload=trials)
else:
	print(trials.best_trial)
	# for t in trials:
	# 	print(t)
	violin_optimizers(trials)
	lr(trials) # lr of adam optimizer
	violin_layers(trials)
	layer_width(trials)

