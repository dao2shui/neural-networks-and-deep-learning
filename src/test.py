# 导入训练数据和测试数据，数据集为mnist，数据集进行分类
from src import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 一个简单的3层随机网络

# 测试1，3层网络，初始输入为784个神经元，30个隐藏神经元，10个输出神经元
import network
net = network.Network([784, 30, 10])
# 梯度下降，30次迭代，小批量数据为10，学习率为3
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# 准确率达到95%

# 测试2，3层网络，初始输入为784个神经元，100个隐藏神经元，10个输出神经元
import network
net = network.Network([784, 100, 10])
# 梯度下降，30次迭代，小批量数据为10，学习率为3
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# 准确率达到96%

# 测试3，3层网络，初始输入为784个神经元，100个隐藏神经元，10个输出神经元
import network
net = network.Network([784, 100, 10])
# 梯度下降，30次迭代，小批量数据为10，学习率为100/0.001
net.SGD(training_data, 30, 10, 100, test_data=test_data)
# net.SGD(training_data, 30, 10, 0.001, test_data=test_data)
# 准确率达到10%，学习率太高或太低都达不到一个很好的学习效果