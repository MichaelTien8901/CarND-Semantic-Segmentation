import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

KEEP_PROB_RATE = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 15
BATCH_SIZE = 12


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def conv1x1(inputs, num_classes, name=None):
    return tf.layers.conv2d(inputs=inputs,
                            filters=num_classes,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            name=name
                            )


def conv_transposed(inputs, num_classes, kernel_size, stride, name=None):
    return tf.layers.conv2d_transpose(inputs=inputs,
                                      filters=num_classes,
                                      kernel_size=(kernel_size, kernel_size),
                                      strides=(stride, stride),
                                      padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      name=name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Full convolutional layer

    l3_resized = conv1x1(vgg_layer3_out, num_classes, name="ss_l3_resized")

    l4_resized = conv1x1(vgg_layer4_out, num_classes, name="ss_l4_resized")
    # 1x1 conv
    l7_flatten = conv1x1(vgg_layer7_out, num_classes, name="ss_l7_resized")
    #    tf.Print(output, [tf.shape(output)])

    l7_decoder = conv_transposed(inputs=l7_flatten, num_classes=num_classes, kernel_size=4, stride=2, name="ss_l7_decoder")

    # add pool4
    l8 = tf.add(l7_decoder, l4_resized, name="ss_l8")
    # decoder, 2x conv7
    l8_decoder = conv_transposed(inputs=l8, num_classes=num_classes, kernel_size=4, stride=2, name="ss_l8_deocder")
    # add pool3
    l9 = tf.add(l8_decoder, l3_resized, name="ss_l9")
    # decode 4x conv7
    output = conv_transposed(inputs=l9, num_classes=num_classes, kernel_size=16, stride=8, name="ss_l9_decoder")

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="ss_logits")
    labels = tf.reshape(correct_label, (-1, num_classes), name="ss_labels_flatten")
    cross_entropy_loss = tf.reduce_mean(
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)), name="ss_cross_entropy_loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="ss_optimizer")
    training_operation = optimizer.minimize(cross_entropy_loss, name="ss_training_op")
    return logits, training_operation, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch + 1))
        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: KEEP_PROB_RATE,
                         learning_rate: LEARNING_RATE}
            _, loss_val = sess.run([train_op, cross_entropy_loss], feed_dict)
            print("Loss: {}".format(loss_val))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes), name="ss_label")
        learning_rate = tf.placeholder(tf.float32, name="ss_learning_rate")

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        layer_out = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, training_op, cross_entropy_loss = optimize(layer_out, correct_label, learning_rate, num_classes)
        saver = tf.train.Saver()
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, training_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
        saver.save(sess, './vgg16_ss')
        print("Model saved")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


def retrain( epochs, batch_size):
    ## num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs2'  #
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    #
    vgg_ss_lebel_tag = 'ss_label:0'
    vgg_ss_learning_rate_tag = 'ss_learning_rate:0'
    # can't get optimizer
    # vgg_ss_training_op_tag = 'ss_training_op:0'
    # vgg_ss_cross_entropy_loss_tag = 'ss_cross_entropy_loss:0'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_ss_logits_tag = 'ss_logits:0'
    vgg_ss_labels_flatten_tag = 'ss_labels_flatten:0'
    with tf.Session() as sess:
        
        # restore previous model
        new_saver = tf.train.import_meta_graph('vgg16_ss.meta')
        graph = tf.get_default_graph()
        # get parameters
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        correct_label = graph.get_tensor_by_name(vgg_ss_lebel_tag)
        learning_rate = graph.get_tensor_by_name(vgg_ss_learning_rate_tag)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        logits = graph.get_tensor_by_name(vgg_ss_logits_tag)
        
        #training_op = graph.get_tensor_by_name(vgg_ss_training_op_tag)
        #cross_entropy_loss = graph.get_tensor_by_name(vgg_ss_cross_entropy_loss_tag)
        # add optimizer
        
        labels = graph.get_tensor_by_name(vgg_ss_labels_flatten_tag)
        cross_entropy_loss = tf.reduce_mean(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(cross_entropy_loss)
        
        sess.run(tf.global_variables_initializer())
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        # retrain more epoches
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)

        new_saver.save(sess, './vgg16_ss_retrain')
        print("Model saved")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)


if __name__ == '__main__':
    run()
    retrain(EPOCHS, BATCH_SIZE)
