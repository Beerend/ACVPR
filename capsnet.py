"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...

Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
Modified by: Frank van der Hoek
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from data_generator import AFLWFaceRegionsSequence

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(
        filters=256,
        kernel_size=9,
        strides=4, # TODO: use 5
        padding='valid',
        activation='relu',
        name='conv1')(
            x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(
        conv1,
        dim_capsule=8,
        n_channels=32,
        kernel_size=9,
        strides=2,
        padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(
        num_capsule=n_class,
        dim_capsule=16,
        routings=routings,
        name='digitcaps')(
            primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    model = models.Model(x, out_caps)

    return model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, train_seq, validation_seq, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(
        log_dir=args.save_dir + '/tensorboard-logs',
        batch_size=args.batch_size,
        histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(
        args.save_dir + '/weights-{epoch:02d}.h5',
        monitor='val_acc',
        save_best_only=True,
        save_weights_only=True,
        verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (args.lr_decay**epoch))

    # compile the model
    model.compile(
        optimizer=optimizers.Adam(lr=args.lr),
        loss=margin_loss,
        metrics={'capsnet': 'accuracy'})
    """
    # Training without data augmentation:
    """
    model.fit_generator(
        generator=train_seq,
        validation_data=validation_seq,
        epochs=args.epochs,
        class_weight={
            0: 1,
            1: 1.8669997421
        },
        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on AFLW.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument(
        '--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument(
        '--lr_decay',
        default=0.9,
        type=float,
        help=
        "The value multiplied by lr at each epoch. Set a larger value for larger epochs"
    )
    parser.add_argument(
        '--lam_recon',
        default=0.392,
        type=float,
        help="The coefficient for the loss of decoder")
    parser.add_argument(
        '-r',
        '--routings',
        default=3,
        type=int,
        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument(
        '--shift_fraction',
        default=0.1,
        type=float,
        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument(
        '--debug', action='store_true', help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument(
        '-t',
        '--testing',
        action='store_true',
        help="Test the trained model on testing dataset")
    parser.add_argument(
        '--digit', default=5, type=int, help="Digit to manipulate")
    parser.add_argument(
        '-w',
        '--weights',
        default=None,
        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    image_size = (224, 224)
    train_regions_csv_file_name = 'regions_train.csv'
    validation_regions_csv_file_name = 'regions_validation.csv'
    path_to_image_folder = '/deepstore/datasets/dmb/Biometrics/Face/aflw/data/flickr'

    train_seq = AFLWFaceRegionsSequence(
        batch_size=args.batch_size,
        regions_csv_file_name=train_regions_csv_file_name,
        path_to_image_folder=path_to_image_folder,
        image_size=image_size)
    validation_seq = AFLWFaceRegionsSequence(
        batch_size=args.batch_size,
        regions_csv_file_name=validation_regions_csv_file_name,
        path_to_image_folder=path_to_image_folder,
        image_size=image_size)

    # define model
    model = multi_gpu_model(CapsNet(
        input_shape=train_seq[0][0].shape[1:], n_class=2, routings=args.routings), gpus=2)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(
            model=model,
            train_seq=train_seq,
            validation_seq=validation_seq,
            args=args)

