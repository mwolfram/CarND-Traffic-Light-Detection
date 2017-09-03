import os.path
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import helper
import glob
import warnings
import sys
import os
from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load vgg model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get required tensors
    vgg_input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    print("vgg3_shape: " + str(vgg_layer3_out.get_shape())) # (?, 56, 56, 256)
    print("vgg4_shape: " + str(vgg_layer4_out.get_shape())) # (?, 28, 28, 512) (?, 14, 14, 512) (?, 7, 7, 512)
    print("vgg7_shape: " + str(vgg_layer7_out.get_shape())) # (?, 1, 1, 4096)

    vgg_layer3_num_outputs = vgg_layer3_out.get_shape()[3]
    vgg_layer4_num_outputs = vgg_layer4_out.get_shape()[3]
    vgg_layer7_num_outputs = vgg_layer7_out.get_shape()[3]

    conv_1x1_layer = tf.layers.conv2d(vgg_layer7_out,
                                      vgg_layer7_num_outputs,
                                      kernel_size=1,
                                      strides=(5, 18),
                                      name="conv_1x1_layer")

    """
    fully_connected_layer = tf.contrib.layers.fully_connected(
        flatten(vgg_layer7_out), # [10, 368640] = 4096*90! wieso, oida?
        num_classes
    )
    """

    # fully connected
    fully_connected_layer = tf.layers.dense(
        conv_1x1_layer, #14400 #3600 #900 too much: x90
        num_classes,
        name="dense_out"
    )





    # layer 3 [array([ 10,  20,  72, 256], dtype=int32)]
                                       # layer 4 [array([ 10,  10,  36, 512], dtype=int32)]
    shapeOp = tf.shape(conv_1x1_layer) #layer 7: [array([  10,    5,   18, 4096], dtype=int32)]

    #return shapeOp, flatten(vgg_layer7_out)
    return shapeOp, fully_connected_layer


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate, can also be a float
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    #nn_last_layer = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return nn_last_layer, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits, shapeOp):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param vgg_keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate, can also be a float
    :param logits: For inferencing during training
    """

    print("Start training")

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Epoch", str(epoch), "|", end="")
        sys.stdout.flush()
        batch_count = 0
        for sample_batch, label_batch in get_batches_fn(batch_size):
            batch_count = batch_count + 1

            # Training
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: sample_batch, correct_label: label_batch, keep_prob: 0.8})

            #print(shape)
            print("=", end="")
            sys.stdout.flush()
        print ("| Loss:", loss)

    if logits is not None:
        saver = tf.train.Saver()
        saver.save(sess, "model")


def run():

    tests_only = os.getenv("TESTS_ONLY", False)
    if tests_only:
        print("TESTS_ONLY environment variable set to True, skipping run.")
        return

    # configuration
    num_classes = 3 # green, nolight, red
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 20
    batch_size = 10
    learning_rate = 0.0005

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function("data", image_shape)

        # load VGG
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # create TF Placeholder for labels
        correct_label = tf.placeholder(tf.int32, (None, num_classes))

        # add layers
        shapeOp, nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # define optimizer
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        import sys
        if len(sys.argv[1:]) > 0:
            saver = tf.train.Saver()
            saver.restore(sess, "model")

        else:
            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate, logits, shapeOp)

        # TODO Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

if __name__ == '__main__':
    run()
