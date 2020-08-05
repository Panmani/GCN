import pickle, datetime
import numpy as np
import  tensorflow as tf
from model import *
import time
import os, sys

# tf.compat.v1.disable_eager_execution()

# model_num = 500

def ABIDE_dataset(graphs, genders, inss, ages, labels, batch_size = 25):
    return tf.data.Dataset.from_tensor_slices(dict(graphs=graphs, genders=genders, inss=inss, ages=ages, y=labels)).repeat().batch(batch_size)

@tf.function
def train_step(model, example, optimizer, cce):
    """
    Trains 'model' on 'example' using 'optimizer'.
    """
    with tf.GradientTape() as tape:
        data = convert_to_model_input(example['graphs']), example['genders'], example['inss'], example['ages'], example['y']
        logits = model(data)
        loss = cce(example['y'], logits)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return logits, loss

if __name__ == '__main__':

    if len(sys.argv) > 1:
        model_idx = sys.argv[1]
        ckpt_dir = os.path.join(ckpt_dir, model_idx)
        print("Checkpoint path:", ckpt_dir)
    else:
        print("Using default checkpoint path:", ckpt_dir)


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


    gcn = GCN()
    dataset = ABIDE_dataset(train_graphs, train_genders, train_inss, train_ages, train_Y, batch_size)
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=gcn, iterator=iterator)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(sys.argv) <= 1:
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    else:
        train_log_dir = 'logs/gradient_tape/' + model_idx + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    start_time = time.time()

    best_val_acc = 0.
    cce = keras.losses.CategoricalCrossentropy()
    for step in range(train_step_num):
        example = next(iterator)
        # tf.summary.trace_on(graph=True, profiler=True)

        logits, loss = train_step(gcn, example, opt, cce)
        mtc = keras.metrics.CategoricalAccuracy()
        mtc.update_state(example['y'], logits)
        acc = mtc.result()
        # with train_summary_writer.as_default():
        #   tf.summary.trace_export(
        #       name="my_func_trace",
        #       step=0,
        #       profiler_outdir='logs/gradient_tape/' + current_time + '/graph')
        ckpt.step.assign_add(1)

        val_logits = gcn( [convert_to_model_input(val_graphs), val_genders, val_inss, val_ages, val_Y], training=False )
        val_logits = gcn( [convert_to_model_input(val_graphs), val_genders, val_inss, val_ages, val_Y], training=False )
        mtc_val = keras.metrics.CategoricalAccuracy()
        mtc_val.update_state(val_Y, val_logits)
        val_acc = float(mtc_val.result())
        with train_summary_writer.as_default():
            tf.summary.scalar('training loss', loss, step=step)
            tf.summary.scalar('training accuracy', acc, step=step)
            tf.summary.scalar('validation accuracy', val_acc, step=step)

        if val_acc > best_val_acc:
            test_logits = gcn( [convert_to_model_input(test_graphs), test_genders, test_inss, test_ages, test_Y], training=False )
            mtc_test = keras.metrics.CategoricalAccuracy()
            mtc_test.update_state(test_Y, test_logits)
            test_acc = float(mtc_test.result())
            with train_summary_writer.as_default():
                tf.summary.scalar('test accuracy', test_acc, step=step)

            save_path = manager.save()
            template = "Step {} --- loss: {}, acc: {} \n\tValidation acc improved from {} to {} \n\tTest acc: {} \n\tModel saved to: {}"
            print( template.format(step, float(loss), float(acc), best_val_acc, val_acc, test_acc, save_path) )
            best_val_acc = val_acc


    print(gcn.summary())
    training_secs = time.time() - start_time
    print("Trained for {} mins {} secs".format(training_secs // 60, int(training_secs % 60)))
    print('Best validation acc:', best_val_acc)
    print('Test acc:', test_acc)
