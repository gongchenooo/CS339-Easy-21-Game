# Easy 21程序结构说明
****  
## 1. 安装所需包
numpy, matplotlib, csv  
## 2. 结构说明
```environment.py```: 

构造easy 21所需要的环境，包括state类和step函数

```q_learning.py```:

q_learning()函数实现了Q-learning的学习方法，main()函数包含基本的训练过程并将结果记录在文件夹QlearningData和QlearningData2中。可以直接运行 python3 q_learning.py进行训练。

```policy_iteration_prepare.py```：

包含分别用Sampling和Markov两种方法计算每个状态stick-value的代码，结果以npy文件的形式记录在PolicyIterationData中。可以直接运行python3 policy_iteration_prepare.py

```policy_iteration.py```:

实现easy 21 game的策略迭代，包含Sampling和Markov两种方法的实现过程，可以直接运行python3 policy_iteration

```test.py```:

包含测试Q-学习和策略迭代得到模型的代码，统计胜率，平率，负率

```plot.py```和```draw.py```:

画图