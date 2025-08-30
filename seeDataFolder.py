#################################################################################
# This file produces in output a tree-like structure of the desired folder.
# The print also shows the number of jpg images inside each leaf folder. This
# is designed to rapidly overview the dataset structure.
# Additionally, in the end also a bar graph of the relative classes, ordered
# by hand is shown through matplotlib.
#################################################################################

import os
import sys
import matplotlib.pyplot as plt

def tree_count_images(start_path='.'):
	x = []
	y = []
	# os.walk generates the file and directory names in a top-down fashion
	for root, dirs, files in os.walk(start_path):
		if "training\\" in root:
			x.append(root.split("\\")[-1])
			y.append(len(files))

		level = root.replace(start_path, '').count(os.sep)
		indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
		image_num = len([f for f in files if f.lower().endswith('.jpg')])
		count_str = f" ({image_num} images)" if image_num > 0 else ""
		print(f'{indent}{os.path.basename(root)}/{count_str}')


	if DATASET_FIRST:
		x = [x[0],x[3], x[4],x[5], x[6],  x[8],x[7], x[1],x[2],x[9],x[10] ]
		y = [y[0],y[3], y[4],y[5], y[6],  y[8],y[7], y[1],y[2],y[9],y[10] ]
		plt.bar(x,y,color=["C0","C0","C1","C1","C2","C3","C3","C4","C4","C6","C7"])
		plt.xticks(fontsize=14, rotation=15)
		plt.yticks(fontsize=16)
		plt.show()
	else:
		print(x,y)
		x = [x[2], x[3] ,x[1], x[0] ]
		plt.bar(x,y,color=["C0","C1","C2","C3"])
		plt.xticks(fontsize=14, rotation=15)
		plt.yticks(fontsize=16)
		plt.show()


DATASET_FIRST = True

if DATASET_FIRST:
	path_to_scan = "data_first"
else:
	path_to_scan = "data_second"

tree_count_images(path_to_scan)

