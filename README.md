# KuiperInfer_rs(自制深度学习推理框架 Rust语言版) 

## 使用的技术和开发环境

这是一个使用 Rust 重构的代码，初学者之作，希望大家给个 star ⭐️！

- 开发语言：Rust
- 数学库：Armadillo
- 加速库：OpenMP



## 快速开始

1. **安装 Rust：** 如果尚未安装 Rust，请参考[Rust 官方网站](https://www.rust-lang.org/)进行安装。
2. **克隆仓库：** 使用以下命令克隆 KuiperInfer_rs 仓库

```bash
git clone  git@github.com:SongQiPing/KuiperInfer_rs.git
```

3. **构建和运行**

   

## 已经支持的算子

- Convolution
- AdaptivePooling
- MaxPooling
- Expression(抽象语法树)
- Flatten(维度展平和变形)
- Sigmoid
- ReLU
- Linear(矩阵相乘)
- Softmax



## 目录

**source**是源码目录

1. **data/** 是张量类Tensor的实现和Tensor初始化方法
2. **layer/** 是算子的实现
3. **parser/** 是Pnnx表达式的解析类
4. **runtime/** 是计算图结构，解析和运行时相关

## 性能测试



## Acknowledgements

Thanks for the following excellent public learning resources.

- [zjhellofss/KuiperInfer](https://github.com/zjhellofss/KuiperInfer) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperInfer?style=social"/> :  带你从零实现一个高性能的深度学习推理库，支持llama 、Unet、Yolov5、Resnet等模型的推理。Implement a high-performance deep learning inference library step by step.

- [zjhellofss/kuiperdatawhale](https://github.com/zjhellofss/kuiperdatawhale) <img src="https://img.shields.io/github/stars/zjhellofss/kuiperdatawhale?style=social"/> :  从零自制深度学习推理框架。

