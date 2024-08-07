import certifi
import cv2
import numpy as np
import gdown
import zipfile

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import keras
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D
import tensorflow.keras.optimizers as optimizers

from imgaug import augmenters
import seaborn as sns


def label_to_numpy(labels):
    final_labels = np.zeros((len(labels), 4))
    for i in range(len(labels)):
        label = labels[i]
        if label == 'Attentive':
          final_labels[i,:] = np.array([1, 0, 0, 0])
        if label == 'DrinkingCoffee':
          final_labels[i,:] = np.array([0, 1, 0, 0])
        if label == 'UsingMirror':
          final_labels[i,:] = np.array([0, 0, 1, 0])
        if label == 'UsingRadio':
          final_labels[i,:] = np.array([0, 0, 0, 1])
    return final_labels


class pkg:
    #### DOWNLOADING AND LOADING DATA
    @staticmethod
    def get_metadata(metadata_path, which_splits = ['train', 'test']):
        '''returns metadata dataframe which contains columns of:
           * index: index of data into numpy data
           * class: class of image
           * split: which dataset split is this a part of?
        '''
        metadata = pd.read_csv(metadata_path)
        keep_idx = metadata['split'].isin(which_splits)
        metadata = metadata[keep_idx]

        # Get dataframes for each class.
        df_coffee_train = metadata[(metadata['class'] == 'DrinkingCoffee') & \
                             (metadata['split'] == 'train')]
        df_coffee_test = metadata[(metadata['class'] == 'DrinkingCoffee') & \
                             (metadata['split'] == 'test')]
        df_mirror_train = metadata[(metadata['class'] == 'UsingMirror') & \
                             (metadata['split'] == 'train')]
        df_mirror_test = metadata[(metadata['class'] == 'UsingMirror') & \
                             (metadata['split'] == 'test')]
        df_attentive_train = metadata[(metadata['class'] == 'Attentive') & \
                             (metadata['split'] == 'train')]
        df_attentive_test = metadata[(metadata['class'] == 'Attentive') & \
                             (metadata['split'] == 'test')]
        df_radio_train = metadata[(metadata['class'] == 'UsingRadio') & \
                             (metadata['split'] == 'train')]
        df_radio_test = metadata[(metadata['class'] == 'UsingRadio') & \
                             (metadata['split'] == 'test')]

        # Get number of items in class with lowest number of images.
        num_samples_train = min(df_coffee_train.shape[0], \
                                df_mirror_train.shape[0], \
                                df_attentive_train.shape[0], \
                                df_radio_train.shape[0])
        num_samples_test = min(df_coffee_test.shape[0], \
                                df_mirror_test.shape[0], \
                                df_attentive_test.shape[0], \
                                df_radio_test.shape[0])

        # Resample each of the classes and concatenate the images.
        metadata_train = pd.concat([df_coffee_train.sample(num_samples_train), \
                              df_mirror_train.sample(num_samples_train), \
                              df_attentive_train.sample(num_samples_train), \
                              df_radio_train.sample(num_samples_train) ])
        metadata_test = pd.concat([df_coffee_test.sample(num_samples_test), \
                              df_mirror_test.sample(num_samples_test), \
                              df_attentive_test.sample(num_samples_test), \
                              df_radio_test.sample(num_samples_test) ])

        metadata = pd.concat( [metadata_train, metadata_test] )
        return metadata

    @staticmethod
    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        '''
        returns images (data), labels from folder of format [image_folder]/[split_name]/[class_name]/
        flattens if flatten option is True
        '''
        # Get dataframes for each class.
        df_coffee_train = metadata[(metadata['class'] == 'DrinkingCoffee') & \
                             (metadata['split'] == 'train')]
        df_coffee_test = metadata[(metadata['class'] == 'DrinkingCoffee') & \
                             (metadata['split'] == 'test')]
        df_mirror_train = metadata[(metadata['class'] == 'UsingMirror') & \
                             (metadata['split'] == 'train')]
        df_mirror_test = metadata[(metadata['class'] == 'UsingMirror') & \
                             (metadata['split'] == 'test')]
        df_attentive_train = metadata[(metadata['class'] == 'Attentive') & \
                             (metadata['split'] == 'train')]
        df_attentive_test = metadata[(metadata['class'] == 'Attentive') & \
                             (metadata['split'] == 'test')]
        df_radio_train = metadata[(metadata['class'] == 'UsingRadio') & \
                             (metadata['split'] == 'train')]
        df_radio_test = metadata[(metadata['class'] == 'UsingRadio') & \
                             (metadata['split'] == 'test')]

        # Get number of items in class with lowest number of images.
        num_samples_train = min(df_coffee_train.shape[0], \
                                df_mirror_train.shape[0], \
                                df_attentive_train.shape[0], \
                                df_radio_train.shape[0])
        num_samples_test = min(df_coffee_test.shape[0], \
                                df_mirror_test.shape[0], \
                                df_attentive_test.shape[0], \
                                df_radio_test.shape[0])

        # Resample each of the classes and concatenate the images.
        metadata_train = pd.concat([df_coffee_train.sample(num_samples_train), \
                              df_mirror_train.sample(num_samples_train), \
                              df_attentive_train.sample(num_samples_train), \
                              df_radio_train.sample(num_samples_train) ])
        metadata_test = pd.concat([df_coffee_test.sample(num_samples_test), \
                              df_mirror_test.sample(num_samples_test), \
                              df_attentive_test.sample(num_samples_test), \
                              df_radio_test.sample(num_samples_test) ])

        metadata = pd.concat( [metadata_train, metadata_test] )

        sub_df = metadata[metadata['split'].isin([split_name])]
        index = sub_df['index'].values
        labels = sub_df['class'].values
        data = all_data[index,:]
        # helpers.plot_one_image(data[0,:])
        """
        print(data.shape)
        for i in range(len(data)):
            data[i, :] = data.reshape(image_shape)
        print(data[0,:].shape)
        """
        if flatten:
            data = data.reshape([-1, np.product(image_shape)])

        """
        plt.imshow(data[0,:])
        plt.show()
        """

        return data, labels

    @staticmethod
    def get_train_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('train', flatten, all_data, metadata, image_shape)

    @staticmethod
    def get_test_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('test', flatten, all_data, metadata, image_shape)

    @staticmethod
    def get_field_data(flatten, all_data, metadata, image_shape):
        return pkg.get_data_split('field', flatten, all_data, metadata, image_shape)

class helpers:
  #### PLOTTING
    @staticmethod
    def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
        '''
        if data is a single image, display that image

        if data is a 4d stack of images, display that image
        '''
        ### cv2.imshow('image', data)


        num_dims   = len(data.shape)
        num_labels = len(labels)
        target_shape = (64,64,3)
        # reshape data if necessary
        if num_dims == 1:
          data = data.reshape(target_shape)
        if num_dims == 2:
          data = data.reshape(np.vstack[-1, image_shape])
        num_dims   = len(data.shape)

        # check if single or multiple images
        if num_dims == 3:
          if num_labels > 1:
            print('Multiple labels does not make sense for single image.')
            return

          label = labels
          if num_labels == 0:
            label = ''
          image = data

        if num_dims == 4:
          image = data[index, :]
          label = labels[index]

        # plot image of interest
        print('Label: %s'%label)
        plt.imshow(image)
        plt.show()

    #### QUERYING AND COMBINING DATA
    @staticmethod
    def get_misclassified_data(data, labels, predictions):
        '''
        Gets the data and labels that are misclassified in a classification task
        Returns:
        -missed_data
        -missed_labels
        -predicted_labels (corresponding to missed_labels)
        -missed_index (indices of items in original dataset)
        '''
        missed_index     = np.where(np.abs(predictions.squeeze() - labels.squeeze()) > 0)[0]
        missed_labels    = labels[missed_index]
        missed_data      = data[missed_index,:]
        predicted_labels = predictions[missed_index]
        return missed_data, missed_labels, predicted_labels, missed_index

    @staticmethod
    def combine_data(data_list, labels_list):
        return np.concatenate(data_list, axis = 0), np.concatenate(labels_list, axis = 0)

    @staticmethod
    def model_to_string(model):
        import re
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        sms = "\n".join(stringlist)
        sms = re.sub('_\d\d\d','', sms)
        sms = re.sub('_\d\d','', sms)
        sms = re.sub('_\d','', sms)
        return sms

    @staticmethod
    def plot_acc(history, ax = None, xlabel = 'Epoch #'):
        history = history.history
        history.update({'epoch':list(range(len(history['val_acc'])))})
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by = 'val_acc', ascending = False).iloc[0]['epoch']

        if not ax:
          f, ax = plt.subplots(1,1)
        sns.lineplot(x = 'epoch', y = 'val_acc', data = history, label = 'Validation', ax = ax)
        sns.lineplot(x = 'epoch', y = 'acc', data = history, label = 'Training', ax = ax)
        ax.axhline(0.25, linestyle = '--',color='red', label = 'Chance')
        ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')
        ax.legend(loc = 1)
        ax.set_ylim([0.4, 1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy (Fraction)')

        plt.show()


class models:
    @staticmethod
    def DenseClassifier(hidden_layer_sizes, nn_params, dropout = 1):
        model = tf.keras.models.Sequential()
        model.add(Flatten(input_shape = nn_params['input_shape']))
        for ilayer in hidden_layer_sizes:
          model.add(Dense(ilayer, activation = 'relu'))
          if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))
        model.compile(loss=nn_params['loss'],
                      optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def CNNClassifier(num_hidden_layers, nn_params, dropout = 1):
        model = tf.keras.models.Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(num_hidden_layers-1):
            model.add(Conv2D(32, (3, 3), padding = 'same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dropout(dropout))

        model.add(Dense(units = 64, activation = 'relu'))

        model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))

        opt = keras.optimizers.rmsprop(learning_rate=1e-4, decay=1e-6)

        model.compile(loss=nn_params['loss'],
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    @staticmethod
    def TransferClassifier(name, nn_params, trainable = True):
        expert_dict = {'VGG16': tf.keras.applications.VGG16,
                       'VGG19': tf.keras.applications.VGG19,
                       'ResNet50': tf.keras.applications.ResNet50,
                       'DenseNet121': tf.keras.applications.DenseNet121}

        expert_conv = expert_dict[name](weights = 'imagenet',
                                                  include_top = False,
                                                  input_shape = nn_params['input_shape'])
        for layer in expert_conv.layers:
            layer.trainable = trainable

        # Create a sequential model and add the pre-trained base model
        expert_model = tf.keras.models.Sequential()
        expert_model.add(expert_conv)
        expert_model.add(GlobalAveragePooling2D())

        expert_model.add(Dense(128, activation = 'relu'))
        expert_model.add(Dropout(0.3))

        expert_model.add(Dense(64, activation = 'relu'))

        expert_model.add(Dense(nn_params['output_neurons'], activation = nn_params['output_activation']))

        expert_model.compile(loss = nn_params['loss'],
                      optimizer = optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                      metrics=['accuracy'])

        return expert_model


if __name__ == "__main__":
    ### defining project variables
    os.environ['SSL_CERT_FILE'] = certifi.where()

    # file variables
    image_data_url       = 'https://drive.google.com/uc?id=1qmTuUyn0525-612yS-wkp8gHB72Wv_XP'
    metadata_url         = 'https://drive.google.com/uc?id=1OfKnq3uIT29sXjWSZqOOpceig8Ul24OW'
    image_data_path      = './raw_data/image_data.npy'
    metadata_path        = './raw_data/metadata.csv'
    image_shape          = (64, 64, 3)

    # neural net parameters
    nn_params = {}
    nn_params['input_shape']       = image_shape
    nn_params['output_neurons']    = 4
    nn_params['loss']              = 'categorical_crossentropy'
    nn_params['output_activation'] = 'softmax'

    ### pre-loading all data of interest
    _all_data = np.load('./raw_data/image_data.npy')
    _metadata = pkg.get_metadata(metadata_path, ['train','test','field'])

    # models with input parameters
    DenseClassifier     = lambda hidden_layer_sizes: models.DenseClassifier(hidden_layer_sizes = hidden_layer_sizes, nn_params = nn_params)
    CNNClassifier       = lambda num_hidden_layers: models.CNNClassifier(num_hidden_layers, nn_params = nn_params)
    TransferClassifier  = lambda name: models.TransferClassifier(name = name, nn_params = nn_params)

    monitor = tf.keras.callbacks.ModelCheckpoint('trained_models/vgg16_5epoch.ckpt', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

    # get a table with information about ALL of our images
    metadata = pkg.get_metadata(metadata_path, ['train','test'])

    # print(metadata.head())

    train_data, train_labels = pkg.get_train_data(flatten = False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
    test_data, test_labels = pkg.get_test_data(flatten = False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)

    image = train_data[0, :]
    image_label = train_labels[0]

    radio_train_data = train_data[train_labels == 'UsingRadio']
    attentive_train_data = train_data[train_labels == 'Attentive']
    coffee_train_data = train_data[train_labels == 'DrinkingCoffee']
    mirror_train_data = train_data[train_labels == 'UsingMirror']

    train_data = train_data.reshape([-1, 64, 64, 3])
    test_data = test_data.reshape([-1, 64, 64, 3])

    train_labels = label_to_numpy(train_labels)
    test_labels = label_to_numpy(test_labels)

    selected_model = models.TransferClassifier("VGG16", nn_params)

    selected_model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels), shuffle=True, callbacks = [monitor])

    # selected_model.save('my_model.h5')

    """
    vgg_expert = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    vgg_model = tf.keras.models.Sequential()

    vgg_model.add(vgg_expert)

    # and then add our own layers on top of it

    vgg_model.add(GlobalAveragePooling2D())

    vgg_model.add(Dense(1024, activation='relu'))

    vgg_model.add(Dropout(0.3))

    vgg_model.add(Dense(512, activation='relu'))

    vgg_model.add(Dropout(0.3))

    vgg_model.add(Dense(4, activation='sigmoid'))

    # finally, we build the vgg model and turn it on so we can use it!

    vgg_model.compile(loss=nn_params['loss'],

                      optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),

                      metrics=['accuracy'])
    vgg_model.fit(train_data, train_labels, epochs=2, validation_data=(test_data, test_labels), shuffle=True,
                  callbacks=[])
    vgg_model.save('my_model.keras')
    """
