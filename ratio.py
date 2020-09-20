import os, sys
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from config import *
import matplotlib.pyplot as plt
from data import *


def load_data(data_root_directory=DATA_dir, left_table_file=left_table_file, matrix_directory=matrices_dir):
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
    for row in tqdm(range(left_table.shape[0])):
        id = str(left_table.loc[row, 'Id'])
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

    input_graphs = np.array(matrices)
    input_genders = np.array(genders)
    input_inss = np.array(inss)
    input_ages = np.array(ages)
    input_ages /= 100.
    input_Y = np.array(labels)

    input_graphs, input_genders, input_inss, input_ages, input_Y = shuffle(input_graphs, input_genders, input_inss, input_ages, input_Y)
    return input_graphs, input_genders, input_inss, input_ages, input_Y

    # train_graphs, val_graphs, test_graphs = \
    #                 input_graphs[:train_size, :, :], \
    #                 input_graphs[train_size:train_size + val_size, :, :], \
    #                 input_graphs[train_size + val_size:, :, :]
    #
    # train_genders, val_genders, test_genders = \
    #                 input_genders[:train_size, :], \
    #                 input_genders[train_size:train_size + val_size, :], \
    #                 input_genders[train_size + val_size:, :]
    #
    # train_inss, val_inss, test_inss = \
    #                 input_inss[:train_size, :], \
    #                 input_inss[train_size:train_size + val_size, :], \
    #                 input_inss[train_size + val_size:, :]
    #
    # train_ages, val_ages, test_ages = \
    #                 input_ages[:train_size], \
    #                 input_ages[train_size:train_size + val_size], \
    #                 input_ages[train_size + val_size:]
    #
    # train_Y, val_Y, test_Y = \
    #                 input_Y[:train_size, :], \
    #                 input_Y[train_size:train_size + val_size, :], \
    #                 input_Y[train_size + val_size:, :]
    #
    # return  train_graphs, val_graphs, test_graphs, \
    #         train_genders, val_genders, test_genders, \
    #         train_inss, val_inss, test_inss, \
    #         train_ages, val_ages, test_ages, \
    #         train_Y, val_Y, test_Y


if __name__ == '__main__':


    input_graphs, input_genders, input_inss, input_ages, input_Y = load_data()
    print(np.sum(input_Y, axis = 0))
    print(np.max(input_graphs))
    print(np.min(input_graphs))
    print(np.max(input_graphs))
    ratio_table = []
    for cur_threshold in np.arange(0, 0.5, 0.01):
        # print(cur_threshold)
        A_backbone = get_backbone_graph(input_graphs, cur_threshold)
        # print(A_backbone)
        cur_ratio = np.sum(A_backbone) / (A_backbone.shape[0] * A_backbone.shape[1])
        # print(np.sum(A_backbone))
        # print(cur_ratio)
        cur_row = [cur_threshold, cur_ratio]
        print(cur_row)
        ratio_table.append(cur_row)

    ratio_table = [[0.0, 0.3912], [0.01, 0.33785], [0.02, 0.29365], [0.03, 0.2558], [0.04, 0.2244], [0.05, 0.1989], [0.06, 0.1774], [0.07, 0.1576], [0.08, 0.1403], [0.09, 0.12545], [0.1, 0.1131], [0.11, 0.1004], [0.12, 0.0897], [0.13, 0.08075], [0.14, 0.0714], [0.15, 0.06445], [0.16, 0.0577], [0.17, 0.053], [0.18, 0.0479], [0.19, 0.04305], [0.2, 0.0388], [0.21, 0.03505], [0.22, 0.0315], [0.23, 0.0282], [0.24, 0.0255], [0.25, 0.02245], [0.26, 0.01985], [0.27, 0.0171], [0.28, 0.0156], [0.29, 0.0135], [0.3, 0.01155], [0.31, 0.00965], [0.32, 0.00805], [0.33, 0.00705], [0.34, 0.0063], [0.35000000000000003, 0.00515], [0.36, 0.00405], [0.37, 0.0036], [0.38, 0.003], [0.39, 0.0024], [0.4, 0.002], [0.41000000000000003, 0.0015], [0.42, 0.00105], [0.43, 0.00075], [0.44, 0.00075], [0.45, 0.00055], [0.46, 0.00045], [0.47000000000000003, 0.00045], [0.48, 0.00035], [0.49, 0.00035]]
    print(ratio_table)
    ratio_table = np.array(ratio_table)
    print(ratio_table.shape)
    plt.plot(ratio_table[:, 0], ratio_table[:, 1])
    plt.show()


    acc_table = np.array([[0.0, 0.6857], [0.05, 0.6857], [0.075, 0.6762], [0.1, 0.7714], [0.125, 0.6952], [0.15, 0.6857], [0.175, 0.6571], [0.2, 0.7048], [0.25, 0.6762], [0.3, 0.6952], [0.35, 0.6857], [0.4, 0.6952], [0.45, 0.6762], [0.5, 0.6762]])
    plt.plot(acc_table[:, 0], acc_table[:, 1])
    plt.show()
