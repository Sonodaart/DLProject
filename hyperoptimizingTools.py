#################################################################################
# This is a collection of function to either hyperoptimize, or show
# plots of an hyperotimization.
#################################################################################

from hyperopt import hp
from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump
import numpy as np


# ============================== Hyperoptimizing functions ==============================
def loss(params, NUM_CLASSES, epochs, train_dataset, validation_dataset):
	model = skinnet(NUM_CLASSES, params)
	print("params: ", params)
	val_acc, val_loss = train(model, train_dataset, validation_dataset, epochs=epochs, verbose=0)
	return {"loss": val_loss, "acc": val_acc, "status": STATUS_OK, "model_type": params["layer_depth"]["type"], "opt":params["opt_config"]["type"]}

def hyperoptimize_all(space, runs, NUM_CLASSES, train_dataset, validation_dataset, trials_preload=None, epochs=50, save=True, verbose=True):
	if trials_preload is None:
		trials = Trials()
	else:
		trials = trials_preload

	lambda_loss = lambda params: loss(params, NUM_CLASSES, epochs, train_dataset, validation_dataset)

	for run in range(runs):
		print(run,range(runs))
		best_params = fmin(
			fn=lambda_loss,
			space=space,
			algo=tpe.suggest,
			max_evals=run+1,
			trials=trials
		)

		if save:
			with open(r"trials_all.pkl", "wb") as file:
				dump(trials,file)

	if verbose:
		print(best_params)
		best_trial = trials.best_trial
		# print(f"Loss: {best_trial["result"]}")

	return trials

# ============================== Plotting functions ==============================
def violin_optimizers(trials):
	results = pd.DataFrame([{
		"loss": trial["result"]["loss"],
		"opt": trial["result"]["opt"]
	} for trial in trials])

	fig, ax = plt.subplots()
	sns.violinplot(x="opt", y="loss", data=results, order=["RMSprop", "Adam"], palette="magma", ax=ax)
	sns.swarmplot(x="opt", y="loss", data=results, color="gray", edgecolor="gray", size=4, ax=ax)
	ax.set_title("Optimizer Choice", fontsize=16, pad=20)
	ax.set_xlabel("Model Configuration", fontsize=12)
	ax.set_ylabel("Validation Loss (Lower is Better)", fontsize=12)
	ax.grid(axis="y", linestyle="--", alpha=0.7)
	fig.tight_layout()
	plt.show()

def violin_layers(trials):
	results = pd.DataFrame([{
		"loss": trial["result"]["loss"],
		"model_type": trial["result"]["model_type"],
		"opt": trial["result"]["opt"]
	} for trial in trials])

	# results = results[results["opt"]=="Adam"] # 

	fig, ax = plt.subplots()
	sns.violinplot(x="model_type", y="loss", data=results, order=["one_layer", "two_layers", "three_layers"], palette="magma", ax=ax)
	sns.swarmplot(x="model_type", y="loss", data=results, color="gray", edgecolor="gray", size=4, ax=ax)
	ax.set_title("Number of layers", fontsize=19, pad=20)
	ax.tick_params(axis='both', which='major', labelsize=13)
	ax.set_xlabel("Model Configuration", fontsize=17)
	ax.set_ylabel("Validation Loss", fontsize=17)
	ax.grid(axis="y", linestyle="--", alpha=0.7)
	fig.tight_layout()
	plt.show()

def layer_width(trials):
	l1 = []
	l2 = []
	l3 = []
	nl = []
	for trial in trials:
		if trial["result"]["model_type"] == "one_layer":
			l1.append(trial["misc"]["vals"]["units1_1L"][0])
			l2.append(np.nan)
			l3.append(np.nan)
			nl.append("one_layer")
		if trial["result"]["model_type"] == "two_layers":
			l1.append(trial["misc"]["vals"]["units1_2L"][0])
			l2.append(trial["misc"]["vals"]["units2_2L"][0])
			l3.append(np.nan)
			nl.append("two_layers")
		if trial["result"]["model_type"] == "three_layers":
			l1.append(trial["misc"]["vals"]["units1_3L"][0])
			l2.append(trial["misc"]["vals"]["units2_3L"][0])
			l3.append(trial["misc"]["vals"]["units3_3L"][0])
			nl.append("three_layers")



	results = pd.DataFrame([{
		"loss": trial["result"]["loss"],
		"opt": trial["result"]["opt"]
	} for trial in trials])
	results["l1"] = np.array(l1)
	results["l2"] = np.array(l2)
	results["l3"] = np.array(l3)
	results["model_type"] = np.array(nl)

	# results = results[results["opt"]=="Adam"] #

	for i in range(1,4):
		fig, ax = plt.subplots(figsize=(9,5))
		ax.set_title(f"Number of nodes in layer {i}", fontsize=19, pad=20)
		ax.tick_params(axis='both', which='major', labelsize=13)
		ax.set_ylabel("Validation Loss", fontsize=17)
		plt.plot(results[f"l{i}"],results["loss"],'.')
		plt.show()


def lr(trials):
	lr = []
	for trial in trials:
		if trial["result"]["opt"] == "Adam": # only adam lr
			print(trial["misc"]["vals"]["learning_rate"])
			lr.append(trial["misc"]["vals"]["learning_rate"][0])
		else:
			lr.append(np.nan)


	results = pd.DataFrame([{
		"loss": trial["result"]["loss"],
		"opt": trial["result"]["opt"]
	} for trial in trials])
	results["lr"] = np.array(lr)

	results = results[results["opt"]=="Adam"]

	fig, ax = plt.subplots(figsize=(9,5))
	ax.set_title("Learning rate", fontsize=19, pad=20)
	ax.tick_params(axis='both', which='major', labelsize=13)
	ax.set_ylabel("Validation Loss", fontsize=17)
	plt.plot(results["lr"],results["loss"],'.')
	plt.xscale("log")
	plt.show()