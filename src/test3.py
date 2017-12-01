# 导入网络类和mnist数据集
import network3
from network3 import ReLU
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
# expanded_training_data, _, _ = network3.load_data_sharaed("../data/mnist_expanded.pkl.gz")
mini_batch_size = 10
# 使用卷积网络优化分类问题

# 测试1，仅适用一个隐藏层，包含100个隐藏层神经元，小批量数据大小为10，
# 60次迭代期，学习速率为0.1
net = Network([FullyConnectedLayer(n_in=784,n_out=100), SoftmaxLayer(n_in=100,n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)


# 测试2，使用一个卷积层，使用5*5局部感受野，20个特征映射，2*2混合窗口，一个全连接隐藏层
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),filter_shape=(20, 1, 5, 5),poolsize=(2, 2)),
               FullyConnectedLayer(n_in=20*12*12, n_out=100),
               SoftmaxLayer(n_in=100, n_out=10)],mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

# 测试3，使用两个卷积层，均为5*5局部感受野，20个特征映射，2*2混合窗口，一个全连接隐藏层
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5),poolsize=(2, 2)),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5),poolsize=(2, 2)),
               FullyConnectedLayer(n_in=40*4*4, n_out=100),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

# 测试4，使用tanh激活函数，两个卷积层，学习率为0.03，正则化参数为0.1
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

# 测试5，使用扩展数据集进行训练，其他参数与测试4相同
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

# 测试6，最后加入两个全连接隐藏层，其他参数与测试5相同
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
               FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

# 测试7，将测试6中添加弃权技术训练全连接层，迭代期变为40次，全连接隐藏层变为1000个神经元
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5),
                             poolsize=(2, 2), activation_fn=ReLU),
               FullyConnectedLayer(n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
               FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
               SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], mini_batch_size)
net.SGD(expanded_training_data, 40, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)