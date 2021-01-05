import os, sys
from tqdm import tqdm
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from config import *
from data import *

def split_by_json(input_ids, input_graphs, input_genders, input_inss, input_ages, input_Y, ids, split_ids, set):
    graphs, genders, inss, ages, Y = [], [], [], [], []
    for id in split_ids[set]:
        idx = ids.index(id)
        graphs.append(input_graphs[idx, :, :, :])
        genders.append(input_genders[idx, :])
        inss.append(input_inss[idx, :])
        ages.append(input_ages[idx])
        Y.append(input_Y[idx, :])
    graphs, genders, inss, ages, Y = \
        np.array(graphs), np.array(genders), np.array(inss), np.array(ages), np.array(Y)
    return graphs, genders, inss, ages, Y

def load_data_by_json(data_root_directory=DATA_dir, left_table_file=left_table_file, matrix_directory=matrices_dir, json_path=json_path):
    """
    Load data from files that are generated by converter.m
    data_root_directory: the root directory where all data files reside
    left_table_file: the file name of the left half of the original table
    matrix_directory: the directory which contains all csv files of matrices,
                file names are the Id entries of their corresponding rows
    """
    left_table = pd.read_csv(os.path.join(data_root_directory, left_table_file))
    print("Left table of shape", left_table.shape, "has been loaded!")
    print("Loading graphs...")
    matrices = []
    labels = []
    inss = []
    genders = []
    ages = []
    ids = []
    for row in tqdm(range(left_table.shape[0])):
        id = str(left_table.loc[row, 'Id'])
        ids.append(id)
        # Read left table
        label_1hot = map_to_onehot(left_table.loc[row, 'label'], LABEL_LIST)
        if label_1hot[0] == 1:
            label_1hot = [1, 0]
        else:
            label_1hot = [0, 1]
        labels.append(label_1hot)
        inss.append(map_to_onehot(left_table.loc[row, 'Ins'], INS_LIST))
        genders.append(map_to_onehot(left_table.loc[row, 'Gender'], GENDER_LIST))
        ages.append(float(left_table.loc[row, 'Age']))
        # Read adjacency matrix
        mtx_path = os.path.join(data_root_directory, matrix_directory, id + ".csv")
        A = np.loadtxt(open(os.path.join(data_root_directory, matrix_directory, id + ".csv"), "r"), delimiter=",", skiprows=0)
        matrices.append(preprocess(A, weight_threshold))

    input_ids = np.array(ids)
    input_graphs = np.array(matrices)
    input_genders = np.array(genders)
    input_inss = np.array(inss)
    input_ages = np.array(ages)
    input_ages /= 100.
    input_Y = np.array(labels)

    with open(json_path, 'r') as json_file:
        split_ids = json.load(json_file)
    train_graphs, train_genders, train_inss, train_ages, train_Y = \
        split_by_json(input_ids, input_graphs, input_genders, input_inss, input_ages, input_Y, ids, split_ids, 'train')
    val_graphs, val_genders, val_inss, val_ages, val_Y = \
        split_by_json(input_ids, input_graphs, input_genders, input_inss, input_ages, input_Y, ids, split_ids, 'val')
    test_graphs, test_genders, test_inss, test_ages, test_Y = \
        split_by_json(input_ids, input_graphs, input_genders, input_inss, input_ages, input_Y, ids, split_ids, 'test')

    return  train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y, input_ids

def get_backbone_graph_by_union(graphs, Y, threshold):
    """
    Get the backbone graph base on graphs from the training set
    """
    A, _ = graphs[:, 0, :, :], graphs[:, 1, :, :]
    A_control = A[Y[:, 0] == 1, :, :]
    A_control = np.mean(A_control, axis = 0)
    A_control[A_control > threshold] = 1.
    A_control[A_control <= threshold] = 0.

    A_asd = A[Y[:, 1] == 1, :, :]
    A_asd = np.mean(A_asd, axis = 0)
    A_asd[A_asd > threshold] = 1.
    A_asd[A_asd <= threshold] = 0.

    A_backbone = A_control + A_asd
    A_backbone[A_backbone > 0] = 1.
    return A_backbone


if __name__ == '__main__':

    train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y, input_ids = load_data_by_json()

    # print(np.sum(get_backbone_graph(train_graphs, weight_threshold)))
    A_backbone = get_backbone_graph(train_graphs, weight_threshold)
    # print(np.sum(A_backbone))
    train_graphs = set_backbone_graph(train_graphs, A_backbone)
    val_graphs = set_backbone_graph(val_graphs, A_backbone)
    test_graphs = set_backbone_graph(test_graphs, A_backbone)

    datasets = train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y

    pickle.dump( datasets, open( pickle_path, "wb" ) )

    train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y = pickle.load( open( pickle_path, "rb" ) )
    print("[Training]   Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        train_graphs.shape, train_genders.shape, train_inss.shape, train_ages.shape, train_Y.shape)
    print("[Validation] Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        val_graphs.shape, val_genders.shape, val_inss.shape, val_ages.shape, val_Y.shape)
    print("[Test]       Graph shape, Gender shape, Ins shape, Ages shape, Y shape: \n\t", \
        test_graphs.shape, test_genders.shape, test_inss.shape, test_ages.shape, test_Y.shape)

    print("[Training]   Class distribution", np.sum(train_Y, axis = 0))
    print("[Validation] Class distribution", np.sum(val_Y, axis = 0))
    print("[Test]       Class distribution", np.sum(test_Y, axis = 0))
    # print(train_X)
    # print(train_X[0].shape)
    # print(np.argmax(train_Y, axis = 1))

    # ===================== UPSAMPLE =====================
    train_graphs, train_genders, train_inss, train_ages, train_Y = \
        upsample(train_graphs, train_genders, train_inss, train_ages, train_Y)

    val_graphs, val_genders, val_inss, val_ages, val_Y = \
        upsample(val_graphs, val_genders, val_inss, val_ages, val_Y)

    test_graphs, test_genders, test_inss, test_ages, test_Y = \
        upsample(test_graphs, test_genders, test_inss, test_ages, test_Y)

    upsampled_data = train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y

    pickle.dump(upsampled_data, open(upsampled_pickle_path, "wb"))

    train_graphs, val_graphs, test_graphs, \
            train_genders, val_genders, test_genders, \
            train_inss, val_inss, test_inss, \
            train_ages, val_ages, test_ages, \
            train_Y, val_Y, test_Y = pickle.load( open( upsampled_pickle_path, "rb" ) )

    print("[Training]   Upsampled class distribution", np.sum(train_Y, axis = 0))
    print("[Validation] Upsampled class distribution", np.sum(val_Y, axis = 0))
    print("[Test]       Upsampled class distribution", np.sum(test_Y, axis = 0))