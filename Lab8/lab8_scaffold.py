

import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell, RNNCell
from tensorflow.contrib.legacy_seq2seq import sequence_loss, rnn_decoder
from mygru import mygru


#
# -------------------------------------------
#
# Global variables

batch_size = 50 # 50 
sequence_length = 50 # 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

# num_layers = 2 # This is no longer being used

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='my_inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

# create a BasicLSTMCell
gru0 = mygru(state_dim);
gru1 = mygru(state_dim);

# use it to create a MultiRNNCell
my_rnn = MultiRNNCell([gru0, gru1], state_is_tuple=True)

# use it to create an initial_state
# note that initial_state will be a *list* of tensors!
initial_state = my_rnn.zero_state(batch_size, tf.float32)

# call seq2seq.rnn_decoder
with tf.variable_scope("encoder") as scope:
    outputs, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, my_rnn)

W = tf.Variable(tf.random_normal([state_dim, vocab_size], stddev=0.02))
b = tf.Variable(tf.random_normal([vocab_size], stddev=0.01))

# transform the list of state outputs to a list of logits.
# use a linear transformation.
logits = [tf.matmul(output, W) +[b] * batch_size for output in outputs]

# call seq2seq.sequence_loss
loss_w = [1.0 for i in range(sequence_length)]
loss = tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, loss_w)

# create a training op using the Adam optimizer
optim = tf.train.AdamOptimizer(0.001).minimize(loss)

# ------------------
# YOUR SAMPLER GRAPH HERE

# Reuse of variables just means you're supposed to reuse the RNN variable

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!



s_batch_size = 1
s_sequence_length = 1

s_in_ph = tf.placeholder( tf.int32, [ s_batch_size ], name='s_inputs' )
s_in_onehot = [tf.one_hot( s_in_ph, vocab_size, name="s_input_onehot" )]


# use it to create an initial_state
# note that initial_state will be a *list* of tensors!
s_initial_state = my_rnn.zero_state(s_batch_size, tf.float32)

# call seq2seq.rnn_decoder
with tf.variable_scope("decoder") as scope:
    s_outputs, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_in_onehot, s_initial_state, my_rnn)

# transform the list of state outputs to a list of logits.
# use a linear transformation.
# s_logits = [tf.matmul(s_output, W) +[b] * s_batch_size for s_output in s_outputs]
# s_probs = tf.nn.softmax(s_logits);
s_probs = tf.matmul(tf.cast(s_outputs[0], tf.float32), W) + b

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        sample = np.argmax( s_probsv[0] )
        # sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

    # print sample( num=60, prime="And " )
    # print sample( num=60, prime="ababab" )
    # print sample( num=60, prime="foo ba" )
    # print sample( num=60, prime="abcdab" )
    print sample( num=60, prime="I " )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
