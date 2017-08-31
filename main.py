import os.path
import tensorflow as tf
import numpy as np
import helper
import glob
import warnings
import sys
import os
#from video import Video
from distutils.version import LooseVersion
import project_tests as tests

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
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    """ From Class """

    """ 1x1 convolution
    The correct use is tf.layers.conv2d(x, num_outputs, 1, 1, weights_initializer=custom_init).
    num_outputs defines the number of output channels or kernels
    The third argument is the kernel size, which is 1.
    The fourth argument is the stride, we set this to 1.
    We use the custom initializer so the weights in the dense and convolutional layers are identical.
    This results in the a matrix multiplication operation that preserves spatial information.
    """

    """ deconvolution
    One possible answer is using tf.layers.conv2d_transpose(x, 3, (2, 2), (2, 2)) to upsample.
    The second argument 3 is the number of kernels/output channels.
    The third argument is the kernel size, (2, 2). Note that the kernel size could also be (1, 1) and the output shape would be the same. However, if it were changed to (3, 3) note the shape would be (9, 9), at least with 'VALID' padding.
    The fourth argument, the number of strides, is how we get from a height and width from (4, 4) to (8, 8). If this were a regular convolution the output height and width would be (2, 2).
    Now that you've learned how to use transposed convolution, let's learn about the third technique in FCNs.
    """

    print("vgg3_shape: " + str(vgg_layer3_out.get_shape())) # (?, 56, 56, 256)
    print("vgg4_shape: " + str(vgg_layer4_out.get_shape())) # (?, 28, 28, 512) (?, 14, 14, 512) (?, 7, 7, 512)
    print("vgg7_shape: " + str(vgg_layer7_out.get_shape())) # (?, 1, 1, 4096)

    vgg_layer3_num_outputs = vgg_layer3_out.get_shape()[3]
    vgg_layer4_num_outputs = vgg_layer4_out.get_shape()[3]
    vgg_layer7_num_outputs = vgg_layer7_out.get_shape()[3]

    # 1x1 convolution
    conv_1x1_layer = tf.layers.conv2d(vgg_layer7_out,
                                      vgg_layer7_num_outputs,
                                      kernel_size=1,
                                      strides=1,
                                      name="conv_1x1_layer")

    print("conv_1x1_layer_shape: " + str(conv_1x1_layer.get_shape()))

    # first deconvolution using conv2d_transpose
    conv_transposed_layer_1 = tf.layers.conv2d_transpose(conv_1x1_layer,
                                                         vgg_layer4_num_outputs,
                                                         kernel_size=(4, 4),
                                                         strides=(2, 2),
                                                         padding="same",
                                                         name="conv_transposed_layer_1")

    print("conv_transposed_layer_1_shape: " + str(conv_transposed_layer_1.get_shape())) # supposed to be (?, ?, ?, 512)

     # skip layer
    skip_layer_1 = tf.add(conv_transposed_layer_1, vgg_layer4_out, name='skip_layer_1')

    # second deconvolution
    conv_transposed_layer_2 = tf.layers.conv2d_transpose(skip_layer_1,
                                                         vgg_layer3_num_outputs,
                                                         kernel_size=(4, 4),
                                                         strides=(2, 2),
                                                         padding="same",
                                                         name='conv_transposed_layer_2')

    print("conv_transposed_layer_2_shape: " + str(conv_transposed_layer_2.get_shape())) # supposed to be (?, ?, ?, 256)

    # skip layer
    skip_layer_2 = tf.add(conv_transposed_layer_2, vgg_layer3_out, name='skip_layer_2')

    # third deconvolution
    conv_transposed_layer_3 = tf.layers.conv2d_transpose(skip_layer_2,
                                                         num_classes,
                                                         kernel_size=(16, 16),
                                                         strides=(8, 8),
                                                         padding="same",
                                                         name='conv_transposed_layer_3')

    print("conv_transposed_layer_3_shape: " + str(conv_transposed_layer_3.get_shape())) # supposed to be (?, ?, ?, ?)

    return conv_transposed_layer_3

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate, can also be a float
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)

def get_iou(logits, image_shape, batch_size):
    unstacked_softmax_logits = tf.unstack(tf.nn.softmax(logits), num=2, axis=1, name='unstack_logits')
    softmax_part = unstacked_softmax_logits[1]
    tf_segmentation = tf.greater(softmax_part, 0.5)
    label_pl = tf.placeholder(tf.float32, [None], name="label_pl")
    iou, iou_op = tf.metrics.mean_iou(label_pl, tf_segmentation, 2)
    return iou, iou_op, label_pl

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits):
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

    # Create IoU ops
    if logits is not None:
        iou, iou_op, label_pl = get_iou(logits, (160, 576), batch_size)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Epoch", str(epoch), "|", end="")
        sys.stdout.flush()
        batch_count = 0
        sum_iou = 0.0
        for sample_batch, label_batch in get_batches_fn(batch_size):
            batch_count = batch_count + 1

            # Training
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: sample_batch, correct_label: label_batch, keep_prob: 0.8})

            # IoU
            if logits is not None:
                label_batch_formatted = label_batch[:,:,:,1].flatten()
                sess.run(tf.local_variables_initializer())
                sess.run(iou_op, {keep_prob: 1.0, input_image: sample_batch, label_pl: label_batch_formatted})
                cur_iou = sess.run(iou, {keep_prob: 1.0, input_image: sample_batch, label_pl: label_batch_formatted})
                sum_iou += cur_iou

            #train_writer.add_summary(summary, i)
            print("=", end="")
            sys.stdout.flush()
        mean_iou = sum_iou / batch_count
        print ("| Loss:", loss, "| IoU:", mean_iou)

    if logits is not None:
        saver = tf.train.Saver()
        saver.save(sess, "model")

tests.test_train_nn(train_nn)

def run():

    tests_only = os.getenv("TESTS_ONLY", False)
    if tests_only:
        print("TESTS_ONLY environment variable set to True, skipping run.")
        return

    # configuration
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 20
    batch_size = 5
    learning_rate = 0.0005

    # check if Kitti dataset is available
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # TODO OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Augment Images for better results
        """ Popular options:
        rotation: random with angle between 0° and 360° (uniform)
        translation: random with shift between -10 and 10 pixels (uniform)
        rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
        flipping: yes or no (bernoulli)
        shearing: random with angle between -20° and 20° (uniform)
        stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)
        """

        print("Creating augmented images...")
        augmented_path = os.path.join(data_dir, 'augmented')
        helper.augment(augmented_path, 'data/data_road/training')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(augmented_path, image_shape)

        # load VGG
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # create TF Placeholder for labels
        correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))

        # add layers
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # define optimizer
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        import sys
        if len(sys.argv[1:]) > 0:
            saver = tf.train.Saver()
            saver.restore(sess, "model")

        else:
            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate, logits)

        # video
        """ How to use:
        1. Uncomment code below and import on top
        2. Get a test video
        3. Define source and target paths below
        4. You'll need to download the moviepy library for video.py to work
        """
        #print("Now working on video...")
        #video_editor = Video("data/test_videos/hart1.mp4", "data/test_videos/hart1_seg2.mp4", sess, logits, vgg_keep_prob, vgg_input, image_shape)
        #video_editor.process_video()

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

if __name__ == '__main__':
    run()
