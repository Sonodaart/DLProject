#################################################################################
# This is a collection of useful functions to analyze the performances of
# a given model. It plots the:
# 	- confusion matrices of the training/validation process
# 	- confusion matrices of the two tasks
# 	- the line graphs shown in the paper
#################################################################################

import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== Confusion matrices ==============================
def confusion_matrix_total(model, ds):
	preds = np.array([])
	ys = np.array([])
	for x,y in ds:
		pred = model(x,training=False).numpy()
		pred = np.argmax(pred,axis=1)
		preds = np.concatenate((preds,pred))
		ys = np.concatenate((ys,y.numpy()))
	NUM_CLASSES = int(np.max(ys))
	cm = confusion_matrix(ys, preds, normalize='pred')
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show()


def confusion_matrix_task(model, taski_validation_dataset, ti_first_index, ti_second_index, task):
	preds = np.array([])
	ys = np.array([])
	for x,y in taski_validation_dataset:
		pred = model(x,training=False).numpy()
		first_pred = np.sum(pred[:,ti_first_index],axis=1)	
		second_pred = np.sum(pred[:,ti_second_index],axis=1)
		# I asserted for 0=kerato, 1=sebo
		pred = (first_pred<second_pred).astype(np.int32)
		preds = np.concatenate((preds,pred))
		ys = np.concatenate((ys,y.numpy()))
	NUM_CLASSES = int(np.max(ys))
	cm = confusion_matrix(ys, preds, normalize='pred')
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=["Malignant","Benign"], yticklabels=["Malignant","Benign"])
	plt.title(f"Task {task}")
	plt.ylabel('Actual')
	plt.xlabel('Predicted')
	plt.show()


# ============================== Line Graphs ==============================
def line_graph_task(model, task1_test_dataset, task2_test_dataset, t1kerato_index, t2melanoma_index):
	def analyze_p(model, p, ds, ti_index):
		preds = np.array([])
		ys = np.array([])
		for x,y in ds:
			pred = model(x,training=False).numpy()
			kerato_pred = np.sum(pred[:,ti_index],axis=1)	
			# I asserted for 0=kerato, 1=sebo
			y = 1-y # 1 if kerato this way
			pred = (kerato_pred>p).astype(np.int32)
			preds = np.concatenate((preds,pred))
			ys = np.concatenate((ys,y.numpy()))
		cm = confusion_matrix(ys, preds, normalize='pred')
		# cm00 is sensitivity, cm11 is specificity
		return cm[0,0], cm[1,1]

	def plot_taski(model, ds, ti_index, task):
		x = []
		y = []
		for i in range(0,100,5):
			x_, y_ = analyze_p(model, i/100, ds, ti_index)
			x.append(x_)
			y.append(y_)
		x = np.array(x)
		y = np.array(y)
		ord = np.argsort(x)
		x = x[ord]
		y = y[ord]

		print("x: ",x)
		print("y: ",y)

		auc = np.trapz(y, x)
		plt.plot(x, y, '.-', label=f'Task {task} (AUC={auc:.3f})')

	plot_taski(model, task1_test_dataset, t1kerato_index, 1)
	plot_taski(model, task2_test_dataset, t2melanoma_index, 2)
	
	for val in [1.0, 0.9, 0.8]:
		plt.plot([val, val], [0, 1], linestyle='--', color='gray', linewidth=1)
		plt.plot([0, 1], [val, val], linestyle='--', color='gray', linewidth=1)

	plt.xlabel("Sensitivity", fontsize=13)
	plt.ylabel("Specificity", fontsize=13)
	plt.xlim([0, 1.07])
	plt.ylim([0, 1.07])
	plt.legend(loc="lower left", frameon=True)
	plt.xlim([0,1.07])
	plt.ylim([0,1.07])
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()


