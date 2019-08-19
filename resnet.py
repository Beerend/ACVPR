import os
import numpy as np
import argparse

from keras import backend as K
from keras import callbacks
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, Dense, Flatten, Activation, BatchNormalization, Add
from keras.utils import multi_gpu_model

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

def Bottleneck(med_filters, out_filters, name, inputs, downsampling=False):
    if(downsampling):
        strides = (2, 2)
        b_identity  = BatchNormalization()(Conv3x3(out_filters, strides, name + '_bi', inputs))
    else:
        strides = (1, 1)
        b_identity  = BatchNormalization()(Conv1x1(out_filters, strides, name + '_bi', inputs))

    b_conv1     = Conv1x1BatchNormReLU(med_filters, (1, 1), name + '_b1', inputs)
    b_conv2     = Conv3x3BatchNormReLU(med_filters, strides, name + '_b2', b_conv1)
    b_conv3     = Conv1x1(out_filters, (1, 1), name + '_b3', b_conv2)
    b_bn3       = BatchNormalization()(b_conv3)
    b_add	= Add()([b_bn3, b_identity])
    b_out       = Activation(activation='relu')(b_add)

    return b_out

def ResNet101(input_shape):
    input_layer = Input(input_shape, name='input_tensor')

    conv1       = Conv7x7BatchNormReLU(64, (2, 2), 'conv1', input_layer)
    maxpool     = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(conv1)

    conv2_1     = Bottleneck(64, 256, 'conv2_1', maxpool)
    conv2_2     = Bottleneck(64, 256, 'conv2_2', conv2_1)
    conv2_3     = Bottleneck(64, 256, 'conv2_3', conv2_2)

    convA_1     = Conv3x3(256, (2, 2), 'convA_1', conv2_3)
    convA_2     = Conv1x1(512, (1, 1), 'convA_2', convA_1)

    conv3_1     = Bottleneck(128, 512, 'conv3_1', conv2_3, True)
    conv3_2     = Bottleneck(128, 512, 'conv3_2', conv3_1)
    conv3_3     = Bottleneck(128, 512, 'conv3_3', conv3_2)
    conv3_4     = Bottleneck(128, 512, 'conv3_4', conv3_3)

    B_0         = Add()([conv3_4, convA_2])
    convB_1     = Conv3x3(512, (2, 2), 'convB_1', B_0)
    convB_2     = Conv1x1(1024, (1, 1), 'convB_2', convB_1)

    conv4_1     = Bottleneck(256, 1024, 'conv4_1', conv3_4, True)
    conv4_2     = Bottleneck(256, 1024, 'conv4_2', conv4_1)
    conv4_3     = Bottleneck(256, 1024, 'conv4_3', conv4_2)
    conv4_4     = Bottleneck(256, 1024, 'conv4_4', conv4_3)
    conv4_5     = Bottleneck(256, 1024, 'conv4_5', conv4_4)
    conv4_6     = Bottleneck(256, 1024, 'conv4_6', conv4_5)
    conv4_7     = Bottleneck(256, 1024, 'conv4_7', conv4_6)
    conv4_8     = Bottleneck(256, 1024, 'conv4_8', conv4_7)
    conv4_9     = Bottleneck(256, 1024, 'conv4_9', conv4_8)
    conv4_10    = Bottleneck(256, 1024, 'conv4_10', conv4_9)
    conv4_11    = Bottleneck(256, 1024, 'conv4_11', conv4_10)
    conv4_12    = Bottleneck(256, 1024, 'conv4_12', conv4_11)
    conv4_13    = Bottleneck(256, 1024, 'conv4_13', conv4_12)
    conv4_14    = Bottleneck(256, 1024, 'conv4_14', conv4_13)
    conv4_15    = Bottleneck(256, 1024, 'conv4_15', conv4_14)
    conv4_16    = Bottleneck(256, 1024, 'conv4_16', conv4_15)
    conv4_17    = Bottleneck(256, 1024, 'conv4_17', conv4_16)
    conv4_18    = Bottleneck(256, 1024, 'conv4_18', conv4_17)
    conv4_19    = Bottleneck(256, 1024, 'conv4_19',  conv4_18)
    conv4_20    = Bottleneck(256, 1024, 'conv4_20', conv4_19)
    conv4_21    = Bottleneck(256, 1024, 'conv4_21', conv4_20)
    conv4_22    = Bottleneck(256, 1024, 'conv4_22', conv4_21)
    conv4_23    = Bottleneck(256, 1024, 'conv4_23', conv4_22)

    C_0         = Add()([conv4_23, convB_2])
    convC_1     = Conv3x3(1024, (2, 2), 'convC_1', C_0)
    convC_2     = Conv1x1(2048, (1, 1), 'convC_2', convC_1)

    conv5_1     = Bottleneck(512, 2048, 'conv5_1', conv4_23, True)
    conv5_2     = Bottleneck(512, 2048, 'conv5_2', conv5_1)
    conv5_3     = Bottleneck(512, 2048, 'conv5_3', conv5_2)

    D_0         = Add()([conv5_3, convC_2])
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
    """
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

    # compile the model
    optimizer, losses, loss_weights, reduce_lr = ResNet101HyperParameters(args.lr)
    model.compile(optimizer = optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'])

    # Training without data augmentation:
    model.fit_generator(
        generator=train_seq,
        validation_data=validation_seq,
        epochs=args.epochs,
        #class_weight={
        #    0: 1,
        #    1: 1.8669997421
        #},
        callbacks=[log, tb, checkpoint, reduce_lr])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)
    return model

def createParser():
    parser = argparse.ArgumentParser(description="ResNet on AFLW.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
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
    model = multi_gpu_model(ResNet101(input_shape=image_size+(3,)), gpus=2)
    #model = ResNet101(input_shape=image_size+(3,))
    model.summary()

    if args.weights is not None:
        model.load_weights(args.weights)
        
    if not args.testing:
        train(model=model, train_seq=train_seq, validation_seq=validation_seq, args=args)
