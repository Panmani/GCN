import pickle, datetime
import numpy as np
import  tensorflow as tf
# from model import *
from data import *
import time
import os, sys
from sklearn import svm
import sklearn

def get_svm_data(graphs, Y):
    H = graphs[:, 1, :, :]
    flatten_H = np.reshape(H, (H.shape[0], -1))
    classes = np.argmax(Y, axis = 1)
    return flatten_H, classes

if __name__ == '__main__':

    train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y = pickle.load( open( upsampled_pickle_path, "rb" ) )
    _, val_graphs, test_graphs, \
            _, val_genders, test_genders, \
            _, val_inss, test_inss, \
            _, val_ages, test_ages, \
            _, val_Y, test_Y = pickle.load( open( pickle_path, "rb" ) )
    print("[Training]   Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        train_graphs.shape, train_genders.shape, train_inss.shape, train_ages.shape, train_Y.shape)
    print("[Validation] Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        val_graphs.shape, val_genders.shape, val_inss.shape, val_ages.shape, val_Y.shape)
    print("[Test]       Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        test_graphs.shape, test_genders.shape, test_inss.shape, test_ages.shape, test_Y.shape)

    train_flatten_H, train_classes = get_svm_data(train_graphs, train_Y)
    train_X = np.concatenate((train_flatten_H, train_genders, train_inss, np.expand_dims(train_ages, axis=-1)), axis = 1)
    val_flatten_H, val_classes = get_svm_data(val_graphs, val_Y)
    val_X = np.concatenate((val_flatten_H, val_genders, val_inss, np.expand_dims(val_ages, axis=-1)), axis = 1)
    test_flatten_H, test_classes = get_svm_data(test_graphs, test_Y)
    test_X = np.concatenate((test_flatten_H, test_genders, test_inss, np.expand_dims(test_ages, axis=-1)), axis = 1)

    # Train SVM
    clf = svm.SVC()
    clf.fit(train_X, train_classes)

    # Evaluate SVM
    train_pred_classes = clf.predict(train_X)
    train_acc = np.sum(train_classes == train_pred_classes) * 1. / train_classes.shape[0]
    print("Training Accuracy:  ", train_acc)
    val_pred_classes = clf.predict(val_X)
    val_acc = np.sum(val_classes == val_pred_classes) * 1. / val_classes.shape[0]
    print("Validation Accuracy:", val_acc)
    pred_label = clf.predict(test_X)
    test_acc = np.sum(test_classes == pred_label) * 1. / test_classes.shape[0]
    print("Test Accuracy:      ", test_acc)

    # Training Accuracy:   0.995
    # Validation Accuracy: 0.7213114754098361
    # Test Accuracy:       0.7142857142857143

    pred_y = np.zeros((pred_label.shape[0], 2))
    for i, label in enumerate(pred_label):
        pred_y[i, label] = 1.
    print(pred_y)

    print("Pred")
    print(pred_label)
    print("True")
    test_label = np.argmax(test_Y, axis = 1)
    print(test_label)
    print(np.sum(train_Y, axis = 0))
    print(np.sum(val_Y, axis = 0))
    print(np.sum(test_Y, axis = 0))

    print("Test Acc = {} = {} / {}".format(np.sum(pred_label == test_label) / test_label.shape[0], np.sum(pred_label == test_label), test_label.shape[0]))

    target_class = 1
    true_positive = np.sum(pred_label[test_label == target_class] == target_class)
    print(true_positive , np.sum(pred_label == target_class))
    print(true_positive , np.sum(test_label == target_class))


    auc = sklearn.metrics.roc_auc_score(test_Y, pred_y)
    precision = sklearn.metrics.precision_score(test_label, pred_label)
    recall = sklearn.metrics.recall_score(test_label, pred_label)
    f1_score = sklearn.metrics.f1_score(test_label, pred_label)

    # TP, FP, TN, FN = perf_measure(test_label, pred_label)
    # print(TP / (TP + FP))
    # print(TP / (TP + FN))
    # specificity = TN / (TN + FP)
    # sensitivity = recall
    # print(specificity)
    # print("=============")
    # print((specificity + sensitivity) / 2)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(test_label, pred_label)
    # print(balanced_acc)
    specificity = balanced_acc * 2 - recall
    print(specificity)
    print('ASD --- Precision = {}  ;  Recall/Sensitivity = {}  ;  Specificity = {}  ;  F1 score = {}  ;  AUC = {}'.format(precision, recall, specificity, f1_score, auc))

    # dot_img_file = 'gcn.png'
    # keras.utils.plot_model(gcn, to_file=dot_img_file, show_shapes=True)
