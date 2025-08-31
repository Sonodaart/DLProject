#################################################################################
# This file, using the written libraries, train the net with the
# specified (like the paper, or personalized) parameters. Then, it perform
# an analysis on the two objective tasks.
#################################################################################

from skinnet import *
from analyzerTools import *

tf.keras.utils.set_random_seed(0)

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


# Paper parameters
params = {
	"layer_depth": {
		"type": "two_layers",
		"units1": 1600,
		"units2": 288,
	},
	"opt_config": {
		"type": "RMSprop",
		"lr": 0.001,
		"decay": 0.9,
		"momentum": 0.9,
		"epsilon": 0.1
	}
}

# handwritten parameters
params = {
	"layer_depth": {
		"type": "two_layers",
		"units1": 1600,
		"units2": 288,
	},
	"opt_config": {
		"type": "Adam",
		"lr": 0.00014157732148629892
	}
}

model = skinnet(NUM_CLASSES, params)
# model = tf.keras.models.load_model("./models/my_model.keras")

train(model, train_dataset, validation_dataset, epochs=50, patience=3)



confusion_matrix_total(model, train_dataset)
confusion_matrix_total(model, validation_dataset)

confusion_matrix_task(model, task1_test_dataset, t1kerato_index, t1sebo_index,1)
# confusion_matrix_task(model, task1_validation_dataset, t1kerato_index, t1sebo_index)
confusion_matrix_task(model, task2_test_dataset, t2melanoma_index, t2nevi_index,2)
# confusion_matrix_task(model, task2_validation_dataset, t2melanoma_index, t2nevi_index)


line_graph_task(model, task1_test_dataset, task2_test_dataset, t1kerato_index, t2melanoma_index)


