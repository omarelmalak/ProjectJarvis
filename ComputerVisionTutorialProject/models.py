import certifi
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
import tensorflow.keras.optimizers as optimizers


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

        if flatten:
            data = data.reshape([-1, np.product(image_shape)])

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


class models:
    @staticmethod
    def ClassifierHandler(name, nn_params, trainable = True):
        expert_dict = {'VGG16': tf.keras.applications.VGG16,
                       'VGG19': tf.keras.applications.VGG19,
                       'ResNet50': tf.keras.applications.ResNet50,
                       'DenseNet121': tf.keras.applications.DenseNet121}

        expert_conv = expert_dict[name](weights = 'imagenet',
                                                  include_top = False,
                                                  input_shape = nn_params['input_shape'])
        for layer in expert_conv.layers:
            layer.trainable = trainable

        expert_model = tf.keras.models.Sequential()
        expert_model.add(expert_conv)
        expert_model.add(GlobalAveragePooling2D())

        expert_model.add(Dense(128, activation = 'relu'))
        expert_model.add(Dropout(0.3))

        expert_model.add(Dense(64, activation = 'relu'))
        expert_model.add(Dropout(0.3))

        expert_model.add(Dense(nn_params['output_neurons'], activation = nn_params['output_activation']))

        expert_model.compile(loss = nn_params['loss'],
                      optimizer = optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                      metrics=['accuracy'])

        return expert_model


if __name__ == "__main__":
    ### defining project variables
    os.environ['SSL_CERT_FILE'] = certifi.where()

    # file variables
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

    monitor = tf.keras.callbacks.ModelCheckpoint('trained_models/gpt_vgg16_10epoch_no_regularizer.ckpt', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

    train_data, train_labels = pkg.get_train_data(flatten = False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
    test_data, test_labels = pkg.get_test_data(flatten = False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)

    train_data = train_data.reshape([-1, 64, 64, 3])
    test_data = test_data.reshape([-1, 64, 64, 3])

    train_labels = label_to_numpy(train_labels)
    test_labels = label_to_numpy(test_labels)

    selected_model = models.ClassifierHandler("VGG16", nn_params)

    selected_model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), shuffle=True, callbacks = [monitor])
