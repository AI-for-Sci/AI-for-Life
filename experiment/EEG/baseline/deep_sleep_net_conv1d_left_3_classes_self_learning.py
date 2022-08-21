
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import time

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
    # x = tf.math.log1p(inputs)
    # Conv1d块 Block 1
    x = layers.Conv1D(filters=64, kernel_size=50, strides=6, padding='same')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    x = layers.MaxPool1D(pool_size=8, strides=8)(x)

    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding='same')(x)
    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding='same')(x)
    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding='same')(x)

    x = layers.MaxPool1D(pool_size=4, strides=4)(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)

    x = layers.Flatten()(x)
    feature = layers.Dropout(rate=0.3)(x)
    output = layers.Dense(num_classes, name='cls', activation='softmax')(x)

    feature_model = tf.keras.Model(inputs, feature, name="Encoder")
    class_model = tf.keras.Model(inputs, output, name="Classification")
    return feature_model, class_model

# 自监督学习损失函数
def simcse_loss(z1, z2, temperature=0.1, LARGE_NUM=10e9):
    # 数据归一化处理
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # 生成监督标签，对角线为1，其他为0
    step_batch_size = tf.shape(z1)[0]
    labels = tf.one_hot(tf.range(step_batch_size), step_batch_size)

    # 进行相似度计算
    logits_aa = tf.matmul(z1, z2, transpose_b=True) / temperature
    # 对角线是1，其他为0，也就是我和我自己相似，和其他不相似
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_aa], 1))
    loss = loss_a
    return loss

# 自监督学习损失函数
def simclr_loss(z1, z2, temperature=0.1, LARGE_NUM=10e9):
    # 数据归一化处理
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    z1_large = z1
    z2_large = z2

    # 生成监督标签，对角线为1，其他为0
    step_batch_size = tf.shape(z1)[0]
    labels = tf.one_hot(tf.range(step_batch_size), step_batch_size * 2)
    masks = tf.one_hot(tf.range(step_batch_size), step_batch_size)

    # 进行相似度计算
    logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / temperature

    # 对角线是1，其他为0，也就是我和我自己相似，和其他不相似
    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)
    return loss


def print_performance(cm):
    W = 0
    N = 1
    REM = 2

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float)  # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float)  # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N: {}".format(tpfn[N]))
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

    y_train_label = []
    for label in y_train:
        if label == 0:
            y_train_label.append(0)
        elif label > 0 and label < 4:
            y_train_label.append(1)
        else:
            y_train_label.append(2)
    y_train = np.array(y_train_label)

    y_val_label = []
    for label in y_valid:
        if label == 0:
            y_val_label.append(0)
        elif label > 0 and label < 4:
            y_val_label.append(1)
        else:
            y_val_label.append(2)
    y_valid = np.array(y_val_label)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))

    train_dataset = tf.data.Dataset.from_tensor_slices(np.vstack([x_train, x_valid])).shuffle(60000).batch(128)

    # 初始化编码器和解码器
    num_classes = 3
    feature_model, class_model = encoder(input_encoder=(3000, 1), num_classes=num_classes)
    # 显示模型结构
    class_model.summary()

    # 优化器和损失函数
    class_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy'))

    # 训练轮次
    epochs = 10
    # 批次数量
    batch_size = 128

    # 优化器
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)


    # ==============================
    #  预训练
    # ==============================
    # 训练函数
    @tf.function
    def train_step(data):
        with tf.GradientTape() as encoder:
            # 生成z_mena, z_var
            hidden_1 = feature_model(data, training=True)
            hidden_2 = feature_model(data, training=True)

            loss = simcse_loss(hidden_1, hidden_2, temperature=0.1, LARGE_NUM=10e9)

        # 模型梯度更新
        gradients_of_enc = encoder.gradient(loss, feature_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients_of_enc, feature_model.trainable_variables))
        return loss


    # ============================================
    # 训练
    # ============================================
    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            for data in train_dataset:
                train_step(data)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))


    train(train_dataset, epochs=3)


    # ==============================
    # 模型训练
    # ==============================
    class_model.fit(x_train,
              y_train,
              shuffle=True,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_valid, y_valid))

    y_pred_val = class_model.predict(x_valid, batch_size=batch_size)
    y_pred_val = np.argmax(y_pred_val, axis=-1)
    valid_cm = confusion_matrix(y_valid, y_pred_val)
    valid_acc = np.mean(y_valid == y_pred_val)
    valid_f1 = f1_score(y_valid, y_pred_val, average="macro")

    # print(valid_cm)
    # print("valid_acc: ", valid_acc)
    # print("valid_f1: ", valid_f1)

    print_performance(valid_cm)

    # Total params: 341701
    # GPU valid_cm  5747m
    # Train times: 19s

    # Sample: 2289
    # W: 319.0
    # N1: 201.0
    # N2: 1222.0
    # N3: 201.0
    # REM: 346.0

    # 5 Classes
    # Precision: [0.976  0.374  0.9596 0.6306 0.6498]
    # Recall: [0.6364 0.4726 0.8159 0.9851 0.8902]
    # F1: [0.7704 0.4176 0.8819 0.7689 0.7512]
    # Overall accuracy: 0.7868
    # Macro - F1 accuracy: 0.718

    # 3 Classes
    # Precision: [0.8703 0.9516 0.7399]
    # Recall: [0.9467 0.9193 0.7977]
    # F1: [0.9069 0.9352 0.7677]
    # Overall accuracy: 0.9048
    # Macro - F1 accuracy: 0.8699


