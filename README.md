# Transfer-Learning-based-on-Resnet34-GTSRB-PyTorch
简介：用Resnet34模型框架 + GTSRB数据集进行迁移学习（第一次手动训练）
## 文件说明：
> `checkpoints`：训练过程中保存的效果较好的模型文件，train_info.txt是训练过程中的部分参数信息、损失值、准确率记录。

> `model_backup`：是目前保存的性能较高的模型文件与模型参数文件，`model_info.txt`记录了模型的最好验证集效果。

> `model_structure`：保存常用的模型结构，其中最终输出的全连接层都进行了修改（43classes）。

> `configs.py`：模型运行的参数配置文件。

> `train_fc.py`：锁定卷积层，只训练最后新添加的全连接层。

> `fine_tune.py`：不锁定参数，微调全部参数。

> `test.py`：测试模型效果并计算准确率。

> `get_model_structure.py`：获取修改全连接层后的模型架构。

> `utils.py`：常用函数的方法合集。


