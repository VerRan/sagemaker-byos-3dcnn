import argparse
import os
import json

import numpy as np 
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


# def save_history(history, result_dir):
#     loss = history.history['loss']
#     acc = history.history['accuracy']
#     val_loss = history.history['val_loss']
#     val_acc = history.history['val_accuracy']
#     nb_epoch = len(acc)

#     with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
#         fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
#         for i in range(nb_epoch):
#             fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
#                 i, loss[i], acc[i], val_loss[i], val_acc[i]))

def _load_training_data(base_dir):
    """Load training data"""
#     x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
#     y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    train = np.load(os.path.join(base_dir, 'train_data.npz'))
    return train

def main():
    
    # 定义超参数   对于sagemaker 通过设置超参数传递进来    
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True) #本地的输出目录
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    #会自动从fit设置的s3路径自动下载到这里以便于模型训练使用
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
#     parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    args = parser.parse_args()

      
    ## 加载训练数据，从训练镜像的设置好的data_dir路径获取数据
    train_data = _load_training_data(args.data_dir)
    X, Y = train_data["X"], train_data["Y"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=43)
    
    nb_classes = args.nclass

    # 定义模型网络
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        X.shape[1:]), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    ## 训练模型
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'test_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'test_3dcnnmodel.hd5'))

#     loss, acc = model.evaluate(X_test, Y_test, verbose=0)
#     print('Test loss:', loss)
#     print('Test accuracy:', acc)
#     save_history(history, args.output)

if __name__ == '__main__':
    main()
