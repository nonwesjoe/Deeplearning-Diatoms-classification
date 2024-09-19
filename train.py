from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

size = 224
num_classes = 1042
epochs = 120
batch_size = 64
image_folder = '/kaggle/input/diatom-datasets/diatom species datasets (seven data augmentation)/diatom species datasets (seven data augmentation)/train'
filepath='/kaggle/working/diatom1042-resnet.keras'
#加载数据
trainds=tf.keras.utils.image_dataset_from_directory(
image_folder,
label_mode='int',
image_size=(size,size),
batch_size=batch_size,
validation_split=0.2,
subset='training',
seed=12)

testds=tf.keras.utils.image_dataset_from_directory(
image_folder,
label_mode='int',
image_size=(size,size),
batch_size=100,
validation_split=0.2,
subset='validation',
seed=12)
#label转换为热编码
def preprocess_labels(images, labels, num_classes=1042):
    labels = tf.one_hot(labels, depth=num_classes)
    return images, labels
trainds = trainds.map(lambda x, y: preprocess_labels(x, y, num_classes=1042))
testds = testds.map(lambda x, y: preprocess_labels(x, y, num_classes=1042))


def resnet_block(input_tensor, filters, kernel_size=3, stride=1, conv_shortcut=False):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor
    
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet34(input_shape=(224, 224, 3), num_classes=10):
    """构建 ResNet34 的模型结构"""
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层和池化层
    x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # 残差块堆叠
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    
    x = resnet_block(x, 128, stride=2, conv_shortcut=True)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    
    x = resnet_block(x, 256, stride=2, conv_shortcut=True)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    
    x = resnet_block(x, 512, stride=2, conv_shortcut=True)
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)
    
    # 全局平均池化和输出层
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# 构建 ResNet34 模型
model = build_resnet34(input_shape=(224, 224, 3), num_classes=num_classes)


# 打印模型结构
model.summary()


# 编译模型
model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 设置模型保存回调
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# 训练模型
history = model.fit(
    trainds,
    epochs=epochs,
    validation_data=testds,
    callbacks=[checkpoint]
)



