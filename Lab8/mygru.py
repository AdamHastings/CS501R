import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops import math_ops


class mygru( RNNCell ):
 
    def __init__( self, num_units):
        self._num_units = num_units
 
    @property
    def state_size(self):
        return self._num_units
 
    @property
    def output_size(self):
        return self._num_units
 
    def __call__( self, inputs, state, scope=None ):
        
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("r"):
                r = sigmoid(_linear([inputs, state], self._num_units, True))
            with tf.variable_scope("z"):
                z = sigmoid(_linear([inputs, state], self._num_units, True))
            with tf.variable_scope("h_tilde"):
                h_tilde = tanh(_linear([inputs, r * state], self._num_units, True))
            new_h = (z * state) + ((1 - z) * h_tilde)
        
        return new_h, new_h