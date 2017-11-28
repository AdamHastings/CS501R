import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from itertools import chain
from numpy.random import randint

# reads file and removes all non a-z characters
def read_file(text_file):
    contents = list(open(source_file, "r"))
    acceptable = range(65, 91) + range(97, 123) + [32]
    contents = [filter(lambda x:ord(x) in acceptable, line).strip().lower() \
                for line in contents]
    contents = [" ".join(re.split(r"\s+", line)) for line in contents] # removes double spaces
    return contents

class Corpus():
    def __init__(self, input_lines, n_train=5000):
        self.SOS = 0
        self.EOS = 1
        self.idx_word, \
        self.word_idx = self.parse_words(input_lines)
        self.n_train = n_train
        
        self.parse_words(input_lines)
        self.corpus_size = len(self.idx_word)
        self.lines = [l.strip().lower() for l in input_lines]
        self.training = [self.sentence_to_index(l) for l in self.lines]
        
    def parse_words(self, lines):
        sls = lambda s: s.strip().lower().split(" ")
        words = ["<SOS>", "<EOS>"] + sorted(set(list( \
                chain(*[sls(l) for l in lines]))))
        idx_word = dict(list(enumerate(words)))
        word_idx = dict(zip(words, list(range(len(words)))))
        
        return idx_word, word_idx
    
    def sentence_to_index(self, s):
        words = s.split(" ")
        indices = [self.word_idx[word] for word in words]
        return indices
    
    def index_to_sentence(self, indices):
        return " ".join(self.idx_word[idx] for idx in indices)

# you will use this class later
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # use rand(n) to get tensors to intitialize your weight matrix and bias tensor 
        # then use Parameter( ) to wrap them as Variables visible to module.parameters()
        # consider xavier dense initialization (the one used without relu)
        
    def forward(self, input_var):
        # standard linear layer, no nonlinearity
        # see torch.matmul
        pass

"""
eencoder topology
-standard GRU topology, see slides for a reveiw
-see if you can figure it out without looking at the decoder's pseuocode below
-context vector is the hidden state of the last time step and last layer
-use nn.GRUCell not nn.GRU
-use zero Variables as the initial hidden states

Notes on RNN workflows in pytorch
-Never use one hot encodings in pytorch. It's programmed to use indexed tensors whenever possible
-pytorch RNNs typically take (batch, seq_len, hidden_dim) tensors
-the result of embedding (batch, seq_len) index tensors of type Long
-but like tensorflow and in this lab RNNCell descendents take 
  (batch, input_dim), (batch, hidden_dim) shaped tensor Variables
"""

class Encoder(nn.Module):
    def __init__(self, hidden_size, source_vocab_size, n_layers=2):
        super(Encoder, self).__init__()
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = source_vocab_size
        self.n_layers = n_layers
        
        #self.embed = nn.Embedding() # needs parameters
        #cells = [nn.GRUCell(*parameters) for _ in n_layers]
        # instantiate a ModuleList as a class member so your GRUCell-s are visible to 
        #   encoder.parameters()
        
    def forward(self, source_variable):        
        # code up a vanilla GRU using nn.GRUCell, return the last output
        pass


"""
Pseudocode for Decoder

--without teacher forcing---
initial_input = Variable ( tensor([[SOS]]) # SOS in Corpus
inputs = [embed(initial_input)]
hidden_states = list of size (n_layers,) of 0-tensors (wrapped as Variables)
for i in 0..len(input_sequence)-1:  
  inputs[i], hidden_states[0]  --first GRUCell--> hidden_states[0]
  hidden_states[0], hidden_states[1] --second GRUCell--> hidden_states[1]
  ... n_layers times    

  apply Linear and torch.SoftMax to hidden state to get the probabilities (should be n_english)
  max index of probabilities --> prediction
  if prediction = target_corpus.EOS:
      break
  create a tensor from prediction and wrap as a Variable
  prediction --embed--> next_input
  append next_input to inputs
  if next_input = EOS:
    break
return probabilities, predictions

-- with teacher forcing --
instead of [embed(initial_input)] use the embedding of source_corpus.SOS
  and the second through last reference words
don't break at EOS
"""


class Decoder(nn.Module):
    def __init__(self, hidden_size, target_vocab_size, n_layers=2, max_target_length=30):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.max_length = self.max_target_length
        
        # initialize n_layers cells of type nn.GRUCell, an nn.Embedding like before
        # initialize a Linear module and nn.ModuleList
        
    def forward(self, context, target_variable=None):
        # if meant to teacher forcing include the target_variable
        use_teacher_forcing = target_variable or None
        
        # return predictions as well for easier sampling


# parameters needed for data processing
n_test = 3000
max_seq_len = 30

## data preprocessing ##
source_file = "data/es.txt"
target_file = "data/en.txt"

source_lines = read_file(source_file)
target_lines = read_file(target_file)

pairs = zip(source_lines, target_lines)
pairs = filter(keep_pair_if, pairs)

n_words = lambda s:len(s.split(" "))
keep_pair_if = lambda pair: max(n_words(pair[0]),n_words(pair[1])) < max_seq_len

training_pairs = filter(keep_pair_if, training_pairs)
source_lines, target_lines = zip(*training_pairs)

source_corpus = Corpus(source_lines)
target_corpus = Corpus(target_lines)
n_spanish = source_corpus.corpus_size
n_english = target_corpus.corpus_size
all_indexed_pairs = zip(source_corpus.training, target_corpus.training)

np.random.seed(2)
test_idc = randint(0, len(training_pairs), n_test)
train_idc = set(np.arange(len(pairs))) - set(test_idc)

# list of 2-string tuples consisting of source / reference sentence pairs
testing_pairs = map(lambda k: pairs[k], test_idc)
training_pairs = map(lambda k: pairs[k], train_idc)

# list of tuples consisting of source / reference sentence pairs 
# but represented by word indexes
train_index_pairs = map(lambda k:all_indexed_pairs[k], train_idc)
test_index_pairs = map(lambda k:all_indexed_pairs[k], test_idc)


# possible hyperparameters
epoch_length = 4000 # can use all sentences
batch_size = 20     # should divide epoch_length
n_layers = 2        # 2 or more recommended
learning_rate = .005
decay_rate = .85 ** (1./epoch_length)
print_every = batch_size  # good practice: should divide epoch_length
n_epochs = 30
hidden_size = 500
teacher_forcing_ratio = .5

encoder = Encoder(hidden_size, n_spanish, n_layers, max_target_length=max_seq_len)
decoder = Decoder(hidden_size, n_english, n_layers, max_target_length=max_seq_len)


def get_loss(output_probs, correct_indices, predicted_indexes):#, predicted_sentence):
    """ 
    params:
      output_probs: a list of Variable (not FloatTensor) with the predicted sequence length
      correct_indices: a list or tensor of type int with the same length, will need
                       to be converted to a Variable before compared to the output_probs
    """
    
    
    # Convert both output_probs and correct_indices to Variables
    # output_probs should have one more dimension than correct_indices
    # Use NLLoss to compute cross entropy without taking softmax twice
    # should return a variable representing the loss    
    # see regularization notes below
    pass


def print_output(teacher_forced, source_indices, predicted_indices, reference_indices, iteration, loss_info=None):
    global source_corpus, target_corpus
    if teacher_forced:
        print("\niteration %d: using teacher forcing" % iteration)
    else:
        print("\niteration %d" % iteration)
    print ("In:       ", source_corpus.index_to_sentence(source_indices))
    print ("Out:      ", target_corpus.index_to_sentence(predicted_indices))
    print ("Reference:", target_corpus.index_to_sentence(predicted_indices))

    

def train(encoder, decoder, training_pairs, testing_pairs, 
                source_corpus, target_corpus, teacher_forcing_ratio, 
                epoch_size, learning_rate, decay, batch_size, print_every):
    """
    You may want to lower the teacher forcing ratio as the number 
      of epochs progresses as it starts to learn word-word connections.
    
    In PyTorch some optimizers don't allow for decaying learning rates
    -Adam does however
    -however initializing new optimizers is trivial
    -You may want to use a learning rate schedule instead of decay
    """
        
    # initialize the optimizer(s) using both the encoder's and decoder's parameters
    
    batched_loss = 0
    for i in range(n_epochs):
        for j in range(epoch_size):
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio

            # consider whether or not to use teacher forcing on printing iterations
            # use_teacher_forcing = use_teacher_forcing or (i % print_every == 0)
            
            source, reference = training_pairs[randint(0, len(training_pairs))]
            # convert source and reference to Tensors then Variables (use type Long not Int)
            
            # run source_var through the encoder
            # run the context vector through the decoder
            # if use_teacher_forcing include the reference variable when calling your decoder
            
            # feed the output probabilities and the reference sentence's indices to the loss
            #   where it will do cross entropy using NLLLoss
            
            # loss = get_loss
            loss.backward()
            
            # implement the batch update as shown in the spec
            
            
            if (j+1) % print_every == 0:
                print_output(use_teacher_forcing, source_idc, predicted, target_idc)
                                
        # run test iteration, print loss, accuracy, perplexity
        
    return encoder, decoder


# use this to print out your final translation sentences 
def sample(encoder, decoder, source_sentences, reference_sentences):#, testing_results = None):
    for source, reference in zip(source_sentences, reference_sentences):
        source = Variable(LongTensor(source), volatile=True)
        # volatile means that the computation graph does not accumulate
        # never use volatile except at inference time, any leaf nodes with 
        #   will cause the computational graph to not accumulate
        # run source through your encoder and decoder and get the predicted sentence
        


# encoder, decoder = train(...)

# print out 100 samples from your test set using sample


# elective, may find this helpful if not using a GPU
def save_weights(encoder, decoder):
    params = list(encoder.named_parameters()) + list(decoder.named_parameters())
    params_np = [(p[0], p[1].data.cpu().numpy()) for p in params]
    for name, param in params_np:
        pass # see np.save(name, param) or pickle

def load_weights(encoder, decoder):
    # load ndarrays from disk, convert them to tensors
    params = list(encoder.named_parameters()) + list(decoder.named_parameters())
    for p, t in zip(params, tensors):
        name, param = p
        param.data.set_(t)
     
"""
if it's not working
-tune hyperparameters
-batch_size, teacher forcing, learning rate are good places to start
-Make sure your teacher forcing implementation is correct
-Try setting max_norm to 1 and scale_grad_by_freq=True when initializing both embeddings
-if scale_grad_by_freq=True concatenate Variables before embedding whenever possible
-Consult Sutskever's 2014 paper

Training philosophy behind vanilla seq2seq nmt systems (see also sutskever, 2014):
-need to learn somewhat one to one word connections first
-hift to learning long term dependencies 
-consider a schedule for learning rate and/or teacher forcing ratio

Saving and restoring weights might speed up your workflow significantly
-might want to checkpoint once you've learned word-word connections

Regularization is super simple in pytorch.
-Be careful with doing dropout on the hidden states between cells
  see https://arxiv.org/pdf/1512.05287.pdf.
-low to moderate dropout after both embeddings may be helpful
"""        