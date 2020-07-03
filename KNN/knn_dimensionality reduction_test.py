from tensorflow.keras import datasets
import tensorflow as tf
import math
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from PIL import Image
#虽然导入了tensorflow，但实际上并没有用到tensorflow的神经网络的框架，只是用它来得到并稍微处理了一下数据集

# 按照四步走来编写代码，四步即 准备数据->搭建网络->训练网络->测试网络
# 准备数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# 训练图像保存在一个uint8 类型的数组中，其形状为(60000, 28, 28)，取值区间为[0, 255]。
# 我们需要将其变换为一个float32 数组，其形状为(60000, 28 * 28)，取值范围为0~1。
train_images = train_images.reshape(60000, 28*28).astype('float32') / 255
test_images = test_images.reshape(10000, 28*28).astype('float32') / 255
pca = PCA(n_components = 100)
pca.fit(train_images) #fit PCA with training data instead of the whole dataset
train_data_pca = pca.transform(train_images)
test_data_pca = pca.transform(test_images)
# 搭建KNN并测试(此处我们用KNN来实现手写识别严格意义上来说，并不是搭建一个网络，这三个步骤都包含在下面)
# 且以下是用minst数据集的训练集和测试集分别来测试这种最简单的KNN算法的准确性，我的测试结果大概是80%\
def showNum(image):
    img = image.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
    return
def showNum10(image):
    img = image.reshape(10, 10)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
    return

def knn_test(test_sum, train_sum,k):
    print("测试KNN算法的准备性")
    accracy_num = 0
    for i in range(test_sum):
        test_data = test_data_pca[i]
        train_data = train_data_pca[0:train_sum, :]
        dist = (np.sqrt(np.sum(np.square(test_data-train_data), axis=1)))
        minsort=dist.argsort()[:k]
        tablesort=train_labels[minsort]
        cout = np.bincount(tablesort)
        coutmax = np.argmax(cout)
        predict = coutmax
        real_data = test_labels[i]
        if predict == real_data:
            accracy_num += 1
        accracy=accracy_num/test_sum
    print('训练集大小：',train_sum,'  测试集大小:',test_sum,'   K=',k,"   准确性:", accracy)
    return accracy

#handwrite()
# 测试
test_sum = 500
train_sum = 50000

#想要测试的内容↓
accracy=np.zeros(50)
ranges=np.array(50)
for i in range(50):
    accracy[i-1]=knn_test(test_sum, 1000+i*1000,3)
plt.plot(ranges*1000+1000, accracy)
plt.show()
