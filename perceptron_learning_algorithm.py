# -*- coding: utf-8 -*-
#!/usr/bin/env python
# File : perceptron_learning_algorithm.py
# Date : 2019/7/6
# Author: leichao
# Email : leichaocn@163.com

"""简述功能.

详细描述.
"""

__filename__ = "perceptron_learning_algorithm.py"
__date__ = 2019 / 7 / 6
__author__ = "leichao"
__email__ = "leichaocn@163.com"

import numpy as np
#
def perceptron_learning_algorithm(X, y):
    w = np.random.rand(3)   # can also be initialized at zero.
    # 获取错分样本
    misclassified_examples = predict(hypothesis, X, y, w)

    # 如果仍然存在错分样本
    while misclassified_examples.any():
        # 从错分样本里拿出一条样本，及对应的target
        x, expected_y = pick_one_from(misclassified_examples, X, y)
        
        # 更新规则：这一步时最核心的！
        w = w + x * expected_y  # update rule
        
        # 再次获取错分样本
        misclassified_examples = predict(hypothesis, X, y, w)

    return w

# 定义分类器
# 注意，这里的x是增广形式，x的第一个元素为偏置，w的第一个元素应为1，
def hypothesis(x, w):
    return np.sign(np.dot(w, x))

# 对于假设的分类器，传入参数、样本，返回错分样本。
def predict(hypothesis_function, X, y, w):
    predictions = np.apply_along_axis(hypothesis_function, 1, X, w)
    misclassified = X[y != predictions]
    return misclassified

# 随机拿出一个样本
def pick_one_from(misclassified_examples, X, y):
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    index = np.where(np.all(X == x, axis=1))
    return x, y[index]

# 生成训练样本，包括正类和负类
def get_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

# 堆起来
def get_dataset_for(X1, y1, X2, y2):
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def get_dataset(get_examples):
    X1, y1, X2, y2 = get_examples()
    X, y = get_dataset_for(X1, y1, X2, y2)
    return X, y

def main():
    """仅用于单元测试"""
    np.random.seed(88)

    X, y = get_dataset(get_training_examples)

    # 把矩阵X转换成增广形式
    X_augmented = np.c_[np.ones(X.shape[0]), X]

    w = perceptron_learning_algorithm(X_augmented, y)

    print(w) # [-44.35244895   1.50714969   5.52834138]

if __name__ == '__main__':
    main()
