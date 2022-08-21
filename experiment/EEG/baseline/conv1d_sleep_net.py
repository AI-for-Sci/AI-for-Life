
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import sys
sys.path.append('../../../')
from experiment.EEG.deepsleep.data_loader import NonSeqDataLoader
from experiment.EEG.deepsleep.sleep_stage import W, N1, N2, N3, REM

def encoder(input_encoder, num_classes=5):
    """
    编码器
    :param input_encoder:
    :return:
    """
    inputs = keras.Input(shape=input_encoder, name='input_layer')
    # Conv1d块 Block 1
    x = layers.Conv1D(filters=64, kernel_size=15, strides=5, padding='same')(inputs)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)

    x = layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.1)(x)
    output = layers.Dense(num_classes, name='cls', activation='softmax')(x)

    model = tf.keras.Model(inputs, output, name="Encoder")
    return model

def print_performance(cm):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision: {}".format(np.around(precision, 4)))
    print("Recall: {}".format(np.around(recall, 4)))
    print("F1: {}".format(np.around(f1, 4)))

    print("Overall accuracy: {}".format(np.around(acc, 4)))
    print("Macro-F1 accuracy: {}".format(np.around(mf1, 4)))

if __name__ == "__main__":

    # 动态分配显存
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    data_dir = '../data/sc_eeg_fpz_cz'
    n_folds = 10
    fold_idx = 1

    # 加载数据
    data_loader = NonSeqDataLoader(
        data_dir=data_dir,
        n_folds=n_folds,
        fold_idx=fold_idx
    )
    x_train, y_train, x_valid, y_valid, _, _ = data_loader.load_train_data()

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)

    # 初始化编码器和解码器
    num_classes = 5
    model = encoder(input_encoder=(3000, 1), num_classes=num_classes)
    # 显示模型结构
    model.summary()

    # 优化器和损失函数
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy'))

    # 训练轮次
    epochs = 10
    # 批次数量
    batch_size = 128

    # ==============================
    # 模型训练
    # ==============================
    model.fit(x_train,
              y_train,
              shuffle=True,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_valid, y_valid))

    y_pred_val = model.predict(x_valid, batch_size=batch_size)
    y_pred_val = np.argmax(y_pred_val, axis=-1)
    valid_cm = confusion_matrix(y_valid, y_pred_val)
    valid_acc = np.mean(y_valid == y_pred_val)
    valid_f1 = f1_score(y_valid, y_pred_val, average="macro")

    # print(valid_cm)
    # print("valid_acc: ", valid_acc)
    # print("valid_f1: ", valid_f1)

    print_performance(valid_cm)

    # Total params: 90885
    # GPU valid_cm  1690m
    # Train times: 14s

    # Sample: 2289
    # W: 319.0
    # N1: 201.0
    # N2: 1222.0
    # N3: 201.0
    # REM: 346.0

    # Precision: [0.9759 0.3968 0.9651 0.7787 0.5285]
    # Recall: [0.7618 0.3731 0.7921 0.9104 0.9364]
    # F1: [0.8556 0.3846 0.8701 0.8394 0.6757]
    # Overall accuracy: 0.7833
    # Macro - F1 accuracy: 0.7251


