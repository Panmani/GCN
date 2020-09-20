import pickle
from tensorflow import keras
from tensorflow.keras import layers
from model import *
from config import *
from train import *
import sklearn.metrics


# def perf_measure(y_true, y_pred):
#     """
#     https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
#     """
#     TP = FP = TN = FN = 0
#
#     for i in range(len(y_pred)):
#         if y_true[i]==y_pred[i]==1:
#            TP += 1
#         if y_pred[i]==1 and y_true[i]!=y_pred[i]:
#            FP += 1
#         if y_true[i]==y_pred[i]==0:
#            TN += 1
#         if y_pred[i]==0 and y_true[i]!=y_pred[i]:
#            FN += 1
#
#     return TP, FP, TN, FN
#

if __name__ == '__main__':

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

    gcn = GCN()
    dataset = ABIDE_dataset(train_graphs, train_genders, train_inss, train_ages, train_Y, batch_size)
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=gcn, iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    pred_y = gcn.predict([convert_to_model_input(test_graphs), test_genders, test_inss, test_ages])
    # print(pred_y)
    pred_label = np.argmax(pred_y, axis = 1)
    print(pred_label)
    print(np.argmax(val_Y, axis = 1))

    print("=====================================================")
    # print(gcn([test_graphs, test_Y]))
    pred_y = gcn.predict([convert_to_model_input(test_graphs), test_genders, test_inss, test_ages])
    pred_label = np.argmax(pred_y, axis = 1)
    test_label = np.argmax(test_Y, axis = 1)

    print("Pred")
    print(pred_label)
    print("True")
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
    print('ASD --- Precision = {}  ;  Recall/Sensitivity = {}  ;  F1 score = {}  ;  Specificity = {}  ;  AUC = {}'.format(precision, recall, f1_score, specificity, auc))

    # dot_img_file = 'gcn.png'
    # keras.utils.plot_model(gcn, to_file=dot_img_file, show_shapes=True)
