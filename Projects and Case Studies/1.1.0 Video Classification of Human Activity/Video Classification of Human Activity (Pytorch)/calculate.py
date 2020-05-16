#Sayantan Dutta
#Calculate the most confused classes


import numpy as np
from helperFunctions import getUCF101

def fetch_combined_confusion_matrix(single_frame_prediction, prediction_3d, class_list, test, num_classes):

    top_1_accuracy = 0.0
    top_5_accuracy = 0.0
    top_10_accuracy = 0.0

    pred1 = np.load(single_frame_prediction)
    pred2 = np.load(prediction_3d)
    comb_pred = (pred1 + pred2) / 2
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    rand_ind = np.random.permutation(len(test[0]))

    for i in range(len(test[0])):
        index = rand_ind[i]

        pred_label = test[1][index]
        comb_pred_label = comb_pred[index]
        pred_sorted = np.argsort(-comb_pred_label)[0:10]
        conf_matrix[pred_label, pred_sorted[0]] += 1

        if pred_label == pred_sorted[0]:
            top_1_accuracy += 1.0
        if np.any(pred_sorted[0:5] == pred_label):
            top_5_accuracy += 1.0
        if np.any(pred_sorted[:]==pred_label):
            top_10_accuracy += 1.0
        
        print('i:%d (%f,%f,%f)'
              % (i, top_1_accuracy / (i+1), top_5_accuracy / (i+1), top_10_accuracy / (i + 1)))
    
    total_examples = np.sum(conf_matrix,axis=1)
    for i in range(num_classes):
        conf_matrix[i,:] = conf_matrix[i,:]/np.sum(conf_matrix[i,:])

    output = np.diag(conf_matrix)
    ind = np.argsort(output)

    class_list_sorted = np.asarray(class_list)
    class_list_sorted = class_list_sorted[ind]
    sorted_results = output[ind]

    for i in range(num_classes):
        print(class_list_sorted[i],sorted_results[i],total_examples[ind[i]])

    return conf_matrix

def fetch_top_confused_classes(confusion_matrix, all_class, num_class=10):
    confusion_matrix_2 = np.copy(confusion_matrix)
    np.fill_diagonal(confusion_matrix_2, 0)
    x = confusion_matrix_2.flatten()
    ind = np.argpartition(x, -num_class)[-num_class:]
    ind = ind[np.argsort(-x[ind])]
    rows, cols = np.unravel_index(ind, confusion_matrix.shape)

    top_10_classes = []
    for row, col in zip(rows, cols):
        top_10_classes.append((all_class[row], all_class[col]))

    return top_10_classes
#------------#

total_classes = 101

data_directory = '/projects/training/bayw/hdf5/'
class_list, train, test = getUCF101(base_directory = data_directory)

single_frame_conf_matrix = np.load("single_frame_confusion_matrix.npy")
top_single_classes = fetch_top_confused_classes(single_frame_conf_matrix, class_list)
print(" Top 10 confused classes for single frame model: \n", top_single_classes)

conf_matrix_3d = np.load("3d_conv_confusion_matrix.npy")
top_3d_classes = fetch_top_confused_classes(conf_matrix_3d, class_list)
print("Top 10 confused classes for 3D model: \n", top_3d_classes)

confusion_matrix_comb = fetch_combined_confusion_matrix('single_frame_prediction_matrix.npy',
                            '3d_conv_prediction_matrix.npy',
                            class_list,
                            test,
                            total_classes
                        )

top_classes_combined = fetch_top_confused_classes(confusion_matrix_comb, class_list)
print(" Top 10 confused classes for combined model: \n", top_classes_combined)