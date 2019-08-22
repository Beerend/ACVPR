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
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from keras.models import Sequential
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
import matplotlib.pyplot as plt
from data_generator import AFLWFaceRegionsSequence
import cv2 as cv
from sklearn.metrics import precision_recall_curve,roc_curve
import csv

K.set_image_data_format('channels_last')


def Baseline(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(256,5,input_shape=input_shape))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(256,5))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(128,5))
    model.add(MaxPooling2D(3))
    model.add(Flatten())
    model.add(Dense(328))
    model.add(Dense(192))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax', name='baseline'))
    return model

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
        loss='categorical_crossentropy',
        metrics={'baseline': 'accuracy'})
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

def test(model,seq, args, seq_name):
    y_pred = []
    y_test = []
    for x,y in seq:
        y_pred.extend(model.predict_on_batch(x))
        y_test.extend(y)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    print('acc', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    fpr, tpr, thresholds = roc_curve(y_test[:,1], y_pred[:,1])
    with open(os.path.join(args.save_dir,seq_name) + '_roc.csv','w') as out:
        w = csv.writer(out)
        w.writerow(['fpr','tpr'])
        for i,_ in enumerate(fpr):
            w.writerow([fpr[i],tpr[i]])

    precision, recall, thresholds = precision_recall_curve(y_test[:,1], y_pred[:,1])
    with open(os.path.join(args.save_dir,seq_name) + '_pr.csv','w') as out:
        w = csv.writer(out)
        w.writerow(['precision','recall'])
        for i,_ in enumerate(precision):
            w.writerow([precision[i],recall[i]])


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on AFLW.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
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
    parser.add_argument('--save_dir', default='./result_baseline')
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
    path_to_image_folder = '/deepstore/datasets/dmb/Biometrics/Face/aflw/data/flickr'

    # define model
    model = multi_gpu_model(Baseline(input_shape=image_size+(3,),num_classes=2), gpus=2)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train_regions_csv_file_name = 'regions_train.csv'
        validation_regions_csv_file_name = 'regions_validation.csv'
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
        train(
            model=model,
            train_seq=train_seq,
            validation_seq=validation_seq,
            args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test_regions_csv_file_name = 'regions_validation.csv'
        test_seq = AFLWFaceRegionsSequence(
            batch_size=args.batch_size,
            regions_csv_file_name=test_regions_csv_file_name,
            path_to_image_folder=path_to_image_folder,
            image_size=image_size)
        test_seq_90_cw = AFLWFaceRegionsSequence(
            batch_size=args.batch_size,
            regions_csv_file_name=test_regions_csv_file_name,
            path_to_image_folder=path_to_image_folder,
            image_size=image_size,
            rotate=cv.ROTATE_90_CLOCKWISE)
        test_seq_90_ccw = AFLWFaceRegionsSequence(
            batch_size=args.batch_size,
            regions_csv_file_name=test_regions_csv_file_name,
            path_to_image_folder=path_to_image_folder,
            image_size=image_size,
            rotate=cv.ROTATE_90_COUNTERCLOCKWISE)
        test_seq_180 = AFLWFaceRegionsSequence(
            batch_size=args.batch_size,
            regions_csv_file_name=test_regions_csv_file_name,
            path_to_image_folder=path_to_image_folder,
            image_size=image_size,
            rotate=cv.ROTATE_180)
        acc_normal = test(model=model, seq=test_seq, args=args, seq_name='normal')
        acc_90_cw = test(model=model, seq=test_seq_90_cw, args=args, seq_name='90_cw')
        acc_90_ccw = test(model=model, seq=test_seq_90_ccw, args=args, seq_name='90_ccw')
        acc_180 = test(model=model, seq=test_seq_180, args=args, seq_name='180')
