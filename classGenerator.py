#################################################################################
# Given a metadata.csv and a row_data/ folder present in the current folder,
# this files uses them to generate and fill directories with the datasets for
# training, validation and testing on the first and second task.
# The variable DATASET_FIRST, if set to True generate the first dataset,
# while if False generate the second.
#################################################################################


import os
import shutil
import pandas as pd
from treelib import Tree
import numpy as np

DATASET_FIRST = True # if False the second settings are applied

df = pd.read_csv('metadata.csv')

# ============================== Tree Construction ==============================
tree = Tree()
# root node
tree.create_node("Skin Disease", "Skin Disease")

depth1 = list(set(df["diagnosis_1"]))
for i in depth1:
	if i is not np.nan:
		tree.create_node(i, i, parent="Skin Disease")

if DATASET_FIRST:
	N_DIAGNOSIS = 3
else:
	N_DIAGNOSIS = 2
for d in range(2,N_DIAGNOSIS+1):
	depth = list(set(df[f"diagnosis_{d}"]))
	for i in depth:
		if i is not np.nan:
			parents = list(set(df[df[f"diagnosis_{d}"]==i][f"diagnosis_{d-1}"]))
			for p in parents:
				tree.create_node(i, i, parent=p)

tree.show()

# ============================== Partitioning Algorithm ==============================

def descendants(tree,node):
	subtree = tree.subtree(node)
	return [n.identifier for n in subtree.all_nodes()]

def numImages(tree,nodes):
	paths = [list(tree.rsearch(n))[:-1][::-1] for n in nodes if n!="Skin Disease"]
	r = 0
	for p in paths:
		_ = df.copy()
		for i in range(len(p)):
			_ = _[ _[f"diagnosis_{i+1}"]==p[i] ]
		if len(p)+1<=N_DIAGNOSIS:
			_ = _[ _[f"diagnosis_{len(p)+1}"].isna() ]
		r += len(_)
	return r

def partitionDiseases(tree,node):
	global MAX_CLASS_SIZE, partition
	desc = descendants(tree,node)
	r = 0
	n = numImages(tree,desc)
	if n<MAX_CLASS_SIZE or len(desc)==1:
		if n>MIN_CLASS_SIZE:
			r += n
			print(n)
			partition.append(node)
	else:
		for c in tree.children(node):
			partitionDiseases(tree, c.identifier)

if DATASET_FIRST:
	MAX_CLASS_SIZE = 1000
else:
	MAX_CLASS_SIZE = 4000
MIN_CLASS_SIZE = 0
partition = []
partitionDiseases(tree, "Skin Disease")
print(partition)
print(len(partition))


# ============================== Creating Dataset ==============================
def fillDataset():
	# creating training dataset
	path = "./data/"

	def mkdir(name):
		try:
			os.mkdir(name)
		except Exception as e:
			pass

	mkdir(path+"training")
	mkdir(path+"validation")

	mkdir(path+"task1")
	mkdir(path+"task1/testing")
	mkdir(path+"task1/testing/kerato")
	mkdir(path+"task1/testing/sebo")
	
	mkdir(path+"task2")
	mkdir(path+"task2/testing")
	mkdir(path+"task2/testing/melanoma")
	mkdir(path+"task2/testing/nevi")

	# Create the directory
	# REMOVE BY HAND ALL FILES IN TRAINING AND VALIDATION
	validation_ages = [30,50,75]
	for sec in ["training","validation"]:
		# create folders
		for p in partition:
			try:
				os.mkdir(path+sec+"/"+p)
			except Exception as e:
				pass

	# fill folders
	train_val_images = []
	for clas in partition:
		print(clas)
		root_path = list(tree.rsearch(clas))[:-1][::-1]
		_ = df.copy()
		for i in range(len(root_path)):
			_ = _[ _[f"diagnosis_{i+1}"]==root_path[i] ]
		tr = 0
		val = 0
		for img in _["isic_id"].sample(frac=1):
			# depending on age and number of already present images, the images are inserted in the directories
			if float(_[_["isic_id"]==img]["age_approx"]) in validation_ages and val<MAX_CLASS_SIZE:
				train_val_images.append(img)
				shutil.copy(f"raw_data\\{img}.jpg",path+"validation/"+root_path[-1]+"/"+img+".jpg")
				val += 1
			elif tr<MAX_CLASS_SIZE:
				train_val_images.append(img)
				shutil.copy(f"raw_data\\{img}.jpg",path+"training/"+root_path[-1]+"/"+img+".jpg")
				tr += 1


	MAX_TEST_CLASS_SIZE = 1_000

	if not DATASET_FIRST:
		# fill dataset for validation of tasks
		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Malignant") & (_["diagnosis_2"]=="Kerato")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_val_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task1/testing/kerato/"+img+".jpg")
				shutil.copy(f"raw_data\\{img}.jpg",path+"task1/testing/kerato/"+img+".jpg")

		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Benign") & (_["diagnosis_2"]=="Benign epidermal proliferations")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_val_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task1/testing/sebo/"+img+".jpg")

		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Malignant") & (_["diagnosis_2"]=="Melanoma")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_val_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task2/testing/melanoma/"+img+".jpg")
		
		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Benign") & (_["diagnosis_2"]=="Benign melanocytic proliferations")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_val_images:
				c += 1
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task2/testing/nevi/"+img+".jpg")


	if DATASET_FIRST:
		# fill dataset for validation of tasks
		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Malignant") & ((_["diagnosis_2"]=="Malignant epidermal proliferations") | (_["diagnosis_2"]=="Malignant adnexal epithelial proliferations - Follicular"))].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task1/testing/kerato/"+img+".jpg")

		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Benign") & (_["diagnosis_2"]=="Benign epidermal proliferations") & ((_["diagnosis_3"]=="Seborrheic keratosis") | (_["diagnosis_3"]=="Pigmented benign keratosis"))].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task1/testing/sebo/"+img+".jpg")

		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Malignant") & (_["diagnosis_2"]=="Malignant melanocytic proliferations (Melanoma)")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task2/testing/melanoma/"+img+".jpg")
		
		_ = df.copy()
		_ = _[(_["diagnosis_1"]=="Benign") & (_["diagnosis_2"]=="Benign melanocytic proliferations")].sample(frac=1)
		c = 0
		for img in _["isic_id"]:
			if img not in train_images:
				c += 1
				# if gathered enough images
				if c>MAX_TEST_CLASS_SIZE:
					break

				shutil.copy(f"raw_data\\{img}.jpg",path+"task2/testing/nevi/"+img+".jpg")



fillDataset()


