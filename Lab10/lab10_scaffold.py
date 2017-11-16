
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize

sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'style.png', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'content.png', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )  # target
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )		# target

def computeGram(v):
	m, n, c = v.get_shape()[1:3]
	v = tf.reshape(v, [m*n, c])
	return tf.transpose(v) * v

def computeContentLoss(F, P):
	#content_loss = np.subtract(target_content, content_acts)
	content_loss = np.subtract(F, P)
	content_loss = np.square(content_loss)
	content_loss = 0.5 * np.sum(content_loss)
	return content_loss

# ops is list of activation nodes for each layer
#
# --- construct your cost function here
#


content_activation = vgg.conv4_2
style_activations = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]

print content_activation
print style_activations

### Content Loss ###
# F = tf.convert_to_tensor(content_acts)
# P = content_activation
# Cast numpy array back into a tensor
# tf. constant , initialize with numpy array.
# if constant doesn't work, use tf.convert to tensor
target_content = tf.constant() # tensor
target_styles = [tf.constant(l) for l in style_acts] #tensor

content_loss = computeContentLoss(target_content, content_acts)


### Style Loss ###
style_G = [computeGram(l) for l in style_activations]
target_style_G = [computeGram(l) for l in target_styles]
# A = 0 # TODO 
M = 0 # TODO height * width
N = 0 # TODO # of channels
# E_l = np.subtract(G - A)
# E_l = np.square(E_l)
# E_l = np.sum(E_l)
# E_l = (1.0 / (4*np.square(N_l)*np.square(M_l))) * E_l

# w_l = 0 # TODO
const = (1.0 / (4*np.square(N)*np.square(M)))
# style_loss = np.matmul(w_l, E_l)
# style_loss = np.sum(style_loss)
style_loss = [const * np.sum(np.subtract(target_style_G - style_G)) ] #for ]
style_loss = np.sum(style_loss) / 5.0

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)
alpha = 1
beta = 1000

Loss = (alpha * content_loss) + (beta * style_loss)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
train_step = tf.train.AdamOptimizer(0.1).minimize(Loss)


# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

# --- place your optimization loop here

EPOCHS = 6000
# No placeholders in this lab
print "ITER\t\tLOSS\t\t\t\tSTYLE_LOSS\t\t\t\tCONTENT LOSS"

for i in range(EPOCHS+1):
	loss, c_loss, s_loss, _ = sess.run([Loss, content_loss, style_loss, train_step])

	if ((i%100) == 0):
		print str(i) + "\t\t" + str(loss) + "\t\t" + str(s_loss) + "\t\t" + str(c_loss)


