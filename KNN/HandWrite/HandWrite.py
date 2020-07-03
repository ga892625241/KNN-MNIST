from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from PIL import Image
#虽然导入了tensorflow，但实际上并没有用到tensorflow的神经网络的框架，只是用它来得到并稍微处理了一下数据集
# 准备数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# 训练图像保存在一个uint8 类型的数组中，其形状为(60000, 28, 28)，取值区间为[0, 255]。
# 我们需要将其变换为一个float32 数组，其形状为(60000, 28 * 28)，取值范围为0~1。
train_images = train_images.reshape(60000, 28*28).astype('float32') / 255
test_images = test_images.reshape(10000, 28*28).astype('float32') / 255
# 且以下是用minst数据集的训练集和测试集分别来测试这种最简单的KNN算法的准确性，我的测试结果大概是80%\

#展示数字
def showNum(image):
    img = image.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
    return

#手写部分
def handwrite():
    #####################↓请将下面的路径设置正确##########################
    image = Image.open('handwrite.png').convert('L') # 用PIL中的Image.open打开图像
    # .convert('L')是将图片灰度化处理，原本是彩色图片，也就是维度是(28,28,3),将其变为(28,28）
    image_arr = np.array(image)  # 转化成numpy数组
    image_arr = np.reshape(image_arr, 28 * 28).astype('float32') / 255
    for i in range(28*28):
        if(image_arr[i]==1):
            image_arr[i]=0
        else:
            image_arr[i]=1
    showNum(image_arr)
    #获取训练集
    train_data = train_images[0:60000, :]
    #获取距离
    dist = (np.sqrt(np.sum(np.square(image_arr - train_data), axis=1)))
    #求距离最近的3个样本标签的众数
    minsort = dist.argsort()[:3]
    tablesort = train_labels[minsort]
    cout = np.bincount(tablesort)
    coutmax = np.argmax(cout)
    #输出结果
    print('手写结果数字结果是：',coutmax)
    return

'''
#KNN_测试部分
def knn_test(test_sum, train_sum,k):
    print("测试KNN算法的准备性")
    accracy_num = 0
    for i in range(test_sum):
        test_data = test_images[i]
        train_data = train_images[0:train_sum, :]
        dist = (np.sqrt(np.sum(np.square(test_data-train_data), axis=1)))
        minsort=dist.argsort()[:k]
        tablesort=train_labels[minsort]
        cout = np.bincount(tablesort)
        coutmax = np.argmax(cout)
        predict = coutmax
        real_data = test_labels[i]
        if predict == real_data:
            accracy_num += 1
        #print("预测:", predict, "实际:", real_data)
        accracy=accracy_num/test_sum
    print("准确性:", accracy)
    return accracy
'''
handwrite()
