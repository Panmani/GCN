import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from config import *


class GraphConv(layers.Layer):
    """
    Graph convolution layer.
    filters: output feature len
    """
    def __init__(self,
                filters,
                **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape=None):
        super(GraphConv, self).build(input_shape)
        self.kernels = self.add_weight('kernels', (input_shape[1][-1], self.filters))

    def call(self, inputs, training=None):
        """
        A: Adjacency Matrix
        A: Feature Matrix
        N = batch size
        n = number of nodes
        """
        A = inputs[0]
        X = inputs[1]
        N, n = A.shape[0], A.shape[1]
        I = tf.linalg.diag(np.array([[1.,] * n] * N, np.float32))
        A_ = A + I
        D_12 = tf.linalg.diag(tf.reduce_sum(A_, axis = 1) ** -0.5)
        DADH = D_12 @ A_ @ D_12 @ X
        H = tf.nn.relu(tf.tensordot(DADH, self.kernels, 1))

        return A, H

    def get_config(self):
        config = super(GraphConv, self).get_config()
        config.update({
            "filters": self.filters,
            "kernels" : self.kernels.numpy(),
        })
        return config


class SAGPool(layers.Layer):
    """
    SAG Pooling
    top_n: the number of nodes kept in the output
    """
    def __init__(self,
                top_n,
                **kwargs):
        super(SAGPool, self).__init__(**kwargs)
        self.top_n = top_n

    def build(self, input_shape=None):
        super(SAGPool, self).build(input_shape)
        self.attention = self.add_weight('attention', (input_shape[1][-1],))

    def call(self, inputs, training=None):
        """
        A: Adjacency Matrix
        A: Feature Matrix
        N = batch size
        n = number of nodes
        """
        A = inputs[0]
        X = inputs[1]
        N, n = A.shape[0], A.shape[1]
        I = tf.linalg.diag(np.array([[1.,] * n] * N, np.float32))
        A_ = A + I
        D_12 = tf.linalg.diag(tf.reduce_sum(A_, axis = 1) ** -0.5)
        DADH = D_12 @ A_ @ D_12 @ X
        Z = tf.nn.tanh(tf.tensordot(DADH, self.attention, 1)) # shape = (N, n)

        idx = tf.argsort(Z, axis = -1)[:, :self.top_n]
        ii = tf.tile(tf.range(N)[:, tf.newaxis], (1, self.top_n))
        coord = tf.stack([ii, idx], axis=-1)
        X_prime = tf.gather_nd(X, coord) # shape = (N, top_n, param_size)
        Z_mask = tf.gather_nd(Z, coord)
        Z_mask_tile = tf.tile(Z_mask[:, :, tf.newaxis], [1, 1, X.shape[-1]])
        X_out = tf.math.multiply(Z_mask_tile, X_prime)

        A_next = tf.gather_nd(A, coord)
        A_next_T = tf.transpose(A_next, [0, 2, 1])
        A_next_T_idx = tf.gather_nd(A_next_T, coord)
        A_out = tf.transpose(A_next_T_idx, [0, 2, 1])

        return A_out, X_out

    def get_config(self):
        config = super(GraphConv, self).get_config()
        config.update({
            "top_n": self.top_n,
            "attention" : self.attention.numpy(),
        })
        return config


class HGPool(layers.Layer):
    """
    Hierarchical Graph Pooling layer.
    """
    def __init__(self,
                top_n,
                **kwargs):
        super(GraphPool, self).__init__(**kwargs)
        self.top_n = top_n

    def call(self, inputs, training=None):
        """
        A shape = (N, n, n)
        X shape = (N, n, n) & X = H from previous layer
        n = number of nodes
        """
        A = inputs[0]
        X = inputs[1]
        N, n = A.shape[0], A.shape[1]
        I = tf.linalg.diag(np.array([[1.,] * n] * N, np.float32))
        D_1 = tf.linalg.diag(tf.reduce_sum(A, axis = 1) ** -1)
        manhattan_dist = (I - D_1 @ A) @ X
        p = tf.reduce_sum(tf.math.abs(manhattan_dist), axis = -1)

        idx = tf.argsort(p, axis = -1)[:, :self.top_n]
        ii = tf.tile(tf.range(N)[:, tf.newaxis], (1, self.top_n))
        coord = tf.stack([ii, idx], axis=-1)
        H = tf.gather_nd(X, coord)
        A_next = tf.gather_nd(A, coord)
        A_next_T = tf.transpose(A_next, [0, 2, 1])
        A_next_T_idx = tf.gather_nd(A_next_T, coord)
        A_next = tf.transpose(A_next_T_idx, [0, 2, 1])

        return A_next, H

    def get_config(self):
        config = super(GraphPool, self).get_config()
        config.update({
            'top_n': self.top_n,
        })
        return config


class GraphConcat(layers.Layer):
    """
    Concatnate graph output with gender, ins, age

    xxx_size: the length of the output vector
        When set to None, the xxx vector is discarded and will not be concatenated
            e.g., gender_size = None, then the gender vector is discarded
        Graph vector cannot be discarded

    has_weights
        == True: input vectors are multiplied with their own weight matrices
                first, and then get concatenated
        == False: input vectors are concatenated directly.
                In this case, the values of xxx_size do not matter unless
                xxx_size is None; when xxx_size is None, the corresponding
                vector is discarded and will not be concatenated

    Two examples of Initializing this layer:
        GraphConcat(graph_size = 15, gender_size = 2, ins_size = 2, age_size = 2)
        GraphConcat(has_weights = False)
    """
    def __init__(self,
                graph_size = 50,
                gender_size = 2,
                ins_size = 18,
                age_size = 1,
                has_weights = True,
                **kwargs):
        super(GraphConcat, self).__init__(**kwargs)
        self.graph_size = graph_size
        self.gender_size = gender_size
        self.ins_size = ins_size
        self.age_size = age_size
        self.has_weights = has_weights

    def build(self, input_shape=None):
        super(GraphConcat, self).build(input_shape)
        if self.has_weights:
            graph_len, gender_len, ins_len, age_len = input_shape[0][1], input_shape[1][1], input_shape[2][1], 1
            self.graph_weights = self.add_weight('graph', [graph_len, self.graph_size])
            if self.gender_size is not None:
                self.gender_weights = self.add_weight('gender', [gender_len, self.gender_size])
            if self.ins_size is not None:
                self.ins_weights = self.add_weight('ins', [ins_len, self.ins_size])
            if self.age_size is not None:
                self.age_weights = self.add_weight('age', [age_len, self.age_size])

    def call(self, inputs, training=None):
        """
        G = GCN flattened
        gender, ins, age: one hot vectors
        """
        G, gender, ins, age = inputs

        out_list = []
        if self.has_weights:
            G_out = tf.tensordot(G, self.graph_weights, 1)
            out_list.append(G_out)
            if self.gender_size is not None:
                gender_out = tf.tensordot(gender, self.gender_weights, 1)
                out_list.append(gender_out)
            if self.ins_size is not None:
                ins_out = tf.tensordot(ins, self.ins_weights, 1)
                out_list.append(ins_out)
            if self.age_size is not None:
                age_out = tf.tensordot(age[:, tf.newaxis], self.age_weights, 1)
                out_list.append(age_out)
            out = tf.nn.relu(tf.concat(out_list, axis = 1))
        else:
            out_list.append(G)
            if self.gender_size is not None:
                out_list.append(gender)
            if self.ins_size is not None:
                out_list.append(ins)
            if self.age_size is not None:
                out_list.append(age[:, tf.newaxis])
            out = tf.concat(out_list, axis = 1)

        return out

    def get_config(self):
        config = super(GraphConcat, self).get_config()
        config.update({
            "graph_size": self.graph_size,
            "gender_size": self.gender_size,
            "ins_size": self.ins_size,
            "age_size": self.age_size,
            "has_weights": self.has_weights,
            "graph_weights" : self.graph_weights.numpy(),
            "gender_weights" : self.gender_weights.numpy(),
            "ins_weights" : self.ins_weights.numpy(),
            "age_weights" : self.age_weights.numpy(),
        })
        return config


class GCN(keras.Model):
    """
    GCN model definition.

    Note: Two consecutive GraphConv must follow this rule:
        the output len of the previous GraphConv == the input len of the next GraphConv
    """

    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.graph_conv_0 = GraphConv(20)
        # self.graph_conv_1 = GraphConv(20)
        # self.graph_pool_0 = SAGPool(100)
        self.graph_conv_2 = GraphConv(10)
        self.graph_pool_1 = SAGPool(10)
        self.flatten = layers.Flatten()
        # self.graph_concat = GraphConcat(graph_size = 15, gender_size = 2, ins_size = 2, age_size = 2)
        self.graph_concat = GraphConcat(has_weights = False)
        self.dense_1 = layers.Dense(2, activation='softmax')

    def net(self, A_X, gender, ins, age):
        out = self.graph_conv_0(A_X)
        # out = self.graph_conv_1(out)
        # out = self.graph_pool_0(out)
        out = self.graph_conv_2(out)
        out = self.graph_pool_1(out)
        out = self.flatten(out[1])
        out = self.graph_concat([out, gender, ins, age])
        out = self.dense_1(out)
        return out

    def call(self, inputs):
        A, X, gender, ins, age = inputs
        logits = self.net([A, X], gender, ins, age)
        return logits

    def predict(self, instances, **kwargs):
        A, X, gender, ins, age = instances
        logits = self.net([A, X], gender, ins, age)
        return logits


# class TestCallback(Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data
#
#     def on_epoch_end(self, epoch, logs={}):
#         x, y = self.test_data
#         loss, acc = self.model.evaluate(x, y, verbose=0)
#         print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
