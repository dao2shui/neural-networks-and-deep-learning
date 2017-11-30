# 导入训练数据和测试数据，数据集为mnist，数据集进行分类
import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
# 使用交叉熵代价函数对权重和偏执进行更新


# 测试1，3层网络，初始输入为784个神经元，30个隐藏神经元，10个输出神经元
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
# 交叉熵梯度下降，30次迭代，小批量数据为10，学习率为0.5
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
# Accuracy on evaluation data: 9540 / 10000

# 测试2，3层网络，初始输入为784个神经元，100个隐藏神经元，10个输出神经元
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
# 交叉熵梯度下降，30次迭代，小批量数据为10，学习率为0.5
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
# Accuracy on evaluation data: 9685 / 10000

# 测试3，3层网络，初始输入为784个神经元，100个隐藏神经元，10个输出神经元，使用前10000幅图像进行训练
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
# 使用前1000幅图像进行训练，400次迭代期，小批量数据为10，学习率为0.5
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True,
        monitor_training_cost=True)
# Accuracy on evaluation data: 8227 / 10000
# import matplotlib.pyplot as plt
# plt.plot(cost)
# plt.show()

# 测试4，3层网络，初始输入为784个神经元，30个隐藏神经元，10个输出神经元，加入正则化
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
# 使用前1000幅图像进行训练，400次迭代期，小批量数据为10，学习率为0.5，正则化参数为0.1
net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda=0.1,
       monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)
# Accuracy on evaluation data: 9540 / 10000
# 使用完整训练集，400次迭代期，小批量数据为10，学习率为0.5，正则化参数为5
net.SGD(training_data, 400, 10, 0.5, evaluation_data=test_data, lmbda=5,
       monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)
# Accuracy on evaluation data: 9672 / 10000

# 测试5，3层网络，初始输入为784个神经元，30个隐藏神经元，10个输出神经元，使用均值为0标准差为1/squat(len(in))的高斯分布，
# 加入正则化
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# 30次迭代期，小批量数据为10，学习率为0.5，正则化参数为0.1,使用validation数据集
net.SGD(training_data, 30, 10, 0.5, evaluation_data=validation_data, lmbda=5.0,
       monitor_evaluation_accuracy=True)
# Accuracy on evaluation data: 9615 / 10000