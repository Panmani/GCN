import pickle
from tensorflow import keras
from tensorflow.keras import layers
from model import *
from config import *
from train import *


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

    print(pred_label)
    print(test_label)
    print(np.sum(train_Y, axis = 0))
    print(np.sum(val_Y, axis = 0))
    print(np.sum(test_Y, axis = 0))

    print(np.sum(pred_label == test_label) / test_label.shape[0], np.sum(pred_label == test_label), test_label.shape[0])


    # dot_img_file = 'gcn.png'
    # keras.utils.plot_model(gcn, to_file=dot_img_file, show_shapes=True)
