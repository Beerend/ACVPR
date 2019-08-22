import os
import argparse
import csv
import numpy as np
import cv2 as cv
from sklearn.metrics import precision_recall_curve,roc_curve

from keras import backend as K
from keras import callbacks
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, Dense, Flatten, Activation, BatchNormalization, Add
#from keras.utils import multi_gpu_model

from data_generator import AFLWFaceRegionsSequence
from utils import plot_log

def Conv1x1(filters, strides, name, inputs):
    return Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False, name=name)(inputs)

def Conv3x3(filters, strides, name, inputs):
    return Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False, name=name)(inputs)

def Conv7x7(filters, strides, name, inputs):
    return Conv2D(filters=filters, kernel_size=(7, 7), strides=strides, padding='same', use_bias=False, name=name)(inputs)

def Conv1x1BatchNormReLU(filters, strides, name, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv1x1(filters, strides, name, inputs)))

def Conv3x3BatchNormReLU(filters, strides, name, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv3x3(filters, strides, name, inputs)))

def Conv7x7BatchNormReLU(filters, strides, name, inputs):
    return Activation(activation='relu')(BatchNormalization()(Conv7x7(filters, strides, name, inputs)))

def Block(out_filters, name, inputs, downsampling=False):
    if(downsampling):
        strides = (2, 2)
        b_identity  = BatchNormalization()(Conv3x3(out_filters, strides, name + '_bi', inputs))
    else:
        strides = (1, 1)
        b_identity  = BatchNormalization()(Conv1x1(out_filters, strides, name + '_bi', inputs))

    b_conv1     = Conv3x3BatchNormReLU(out_filters, strides, name + '_b1', inputs)
    b_conv2     = Conv3x3(out_filters, (1, 1), name + '_b2', b_conv1)
    b_bn2       = BatchNormalization()(b_conv2)
    b_add       = Add()([b_bn2, b_identity])
    b_out       = Activation(activation='relu')(b_add)

    return b_out

def ResNet101(input_shape):
    input_layer = Input(input_shape, name='input_tensor')

    conv1       = Conv7x7BatchNormReLU(64, (2, 2), 'conv1', input_layer)
    maxpool     = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(conv1)

    conv2_1     = Block(64, 'conv2_1', maxpool)
    conv2_2     = Block(64, 'conv2_2', conv2_1)

    convA_1     = Conv3x3(64, (2, 2), 'convA_1', conv2_2)
    convA_2     = Conv1x1(128, (1, 1), 'convA_2', convA_1)

    conv3_1     = Block(128, 'conv3_1', conv2_2, True)
    conv3_2     = Block(128, 'conv3_2', conv3_1)

    B_0         = Add()([conv3_2, convA_2])
    convB_1     = Conv3x3(128, (2, 2), 'convB_1', B_0)
    convB_2     = Conv1x1(256, (1, 1), 'convB_2', convB_1)

    conv4_1     = Block(256, 'conv4_1', conv3_2, True)
    conv4_2     = Block(256, 'conv4_2', conv4_1)

    C_0         = Add()([conv4_2, convB_2])
    convC_1     = Conv3x3(256, (2, 2), 'convC_1', C_0)
    convC_2     = Conv1x1(512, (1, 1), 'convC_2', convC_1)

    conv5_1     = Block(512, 'conv5_1', conv4_2, True)
    conv5_2     = Block(512, 'conv5_2', conv5_1)

    D_0         = Add()([conv5_2, convC_2])
    avgpool     = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(D_0)
    flatten     = Flatten(name='flatten')(avgpool)

    fc_detect   = Dense(512, activation='relu', name='fc_detect')(flatten)
    #fc_landmks  = Dense(512, activation = 'relu', name = 'fc_landmks')(flatten)
    #fc_vis      = Dense(512, activation = 'relu', name = 'fc_vis')(flatten)
    #fc_pose     = Dense(512, activation = 'relu', name = 'fc_pose')(flatten)
    #fc_gender   = Dense(512, activation = 'relu', name = 'fc_gender')(flatten)

    out_detect  = Dense(2, activation = 'softmax', name = 'out_detect')(fc_detect)
    #out_landmks = Dense(42, activation = 'softmax', name = 'out_landmks')(fc_landmks)
    #out_vis     = Dense(21, activation = 'sigmoid', name = 'out_vis')(fc_vis)
    #out_pose    = Dense(3, activation = 'softmax', name = 'out_pose')(fc_pose)
    #out_gender  = Dense(2, activation = 'softmax', name = 'out_gender')(fc_gender)

    model = Model(inputs = input_layer, outputs = out_detect)
    #model = Model(inputs = input_layer, outputs = [out_detect, out_landmks, out_vis, out_pose, out_gender])
    return model

def ResNet101HyperParameters(lr):
    optimizer = RMSprop(lr = lr, rho = 0.9, epsilon = None, decay = 0.0)

    losses = {'out_detect': 'categorical_crossentropy'}
    '''losses = {
        "out_detect"    : "sparse_categorical_crossentropy",
        "out_landmks"   : "mean_squared_error",
        "out_vis"       : "mean_squared_error",
        "out_pose"      : "mean_squared_error",
        "out_gender"    : "binary_crossentropy"
    }'''

    loss_weights = {'out_detect': 1.0}
    '''loss_weights = {
        "out_detect"    : 1.0,
        "out_landmks"   : 1.0,
        "out_vis"       : 1.0,
        "out_pose"      : 1.0,
        "out_gender"    : 1.0
    }'''
        
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 3, min_lr = 0.00001)

    return optimizer, losses, loss_weights, reduce_lr

def train(model, train_seq, validation_seq, args):
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

    # compile the model
    optimizer, losses, loss_weights, reduce_lr = ResNet101HyperParameters(args.lr)
    model.compile(optimizer = optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'])

    # Training without data augmentation:
    model.fit_generator(
        generator=train_seq,
        validation_data=validation_seq,
        epochs=args.epochs,
        class_weight={
            0: 1,
            1: 1.8669997421
        },
        callbacks=[log, tb, checkpoint, reduce_lr])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)
    return model

def test(model, seq, args, seq_name):
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

def createParser():
    parser = argparse.ArgumentParser(description="ResNet on AFLW.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.00001, type=float,
        help="Initial learning rate")
    parser.add_argument('--debug', action='store_true',
        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result_resnet')
    parser.add_argument('-t', '--testing', action='store_true',
        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
        help="The path of the saved weights. Should be specified when testing")

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    K.set_image_data_format('channels_last')

    args = createParser()

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
    #model = multi_gpu_model(ResNet101(input_shape=image_size+(3,)), gpus=2)
    model = ResNet101(input_shape=image_size+(3,))
    model.summary()

    if args.weights is not None:
        model.load_weights(args.weights)
        
    if not args.testing:
        train(model=model, train_seq=train_seq, validation_seq=validation_seq, args=args)
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
