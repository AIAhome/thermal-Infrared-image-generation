# thermal-Infrared-image-generation
## 简介
热图像具有有用的判别特性。具体而言，温暖的物体（即人类、动物、热车等）往往是感兴趣的对象。

不幸的是，热像仪：

- 很贵
- 视野狭窄
- 不足以完成所有感知任务（例如，无法阅读交通信号灯）

由于这些原因，我们常使用 RGB 相机进行语义分割等感知任务。

**在这项工作，我们将实现从 RGB 图像转换为热图像**
## 算法实现
### 数据处理
见`process.py`,将原始图片resize为256x256
### 评价指标
FID
### 数据集
下载地址：[Download Free ADAS Dataset v2 - Teledyne FLIR](https://adas-dataset-v2.flirconservator.com/#downloadguide) 

总共包含 9，711 张热成像图像和 9，233 张 RGB 训练/验证**图像**，并带有 90%/10% 的训练/评估分割。验证**视频**共计 7，498 帧。3，749 个热/RGB 视频对均由一系列以每秒 30 帧 （FPS） 捕获的连续帧组成。一个频谱中的每个视频帧都映射到另一个频谱中的时间同步帧对，在文件中指定。此数据集中的素材是在各种位置收集的，包括各种照明/天气条件。
## 参考资料

一个收集了大量 GAN 论文的 Github 项目，并且根据应用方向划分论文：

- [AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers)

以及 3 个复现多种 GANs 模型的 github 项目，分别是目前主流的三个框架，TensorFlow、PyTorch 和 Keras：

- [tensorflow-GANs](https://github.com/TwistedW/tensorflow-GANs)：TensorFlow 版本
- [Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)：PyTorch 版本
- [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)：Keras 版本
## 文献整理
### 1. Generative Adversarial Networks

论文名称：Generative Adversarial Nets

论文地址: https://arxiv.org/abs/1406.2661

“GAN之父” Ian Goodfellow 发表的第一篇提出 GAN 的论文，这应该是任何开始研究学习 GAN 的都该阅读的一篇论文，它提出了 GAN 这个模型框架，讨论了非饱和的损失函数，然后对于最佳判别器(optimal discriminator)给出其导数，然后进行证明；最后是在 Mnist、TFD、CIFAR-10 数据集上进行了实验。

### 2. Conditional GANs

论文名称：Conditional Generative Adversarial Nets

论文地址：https://arxiv.org/abs/1411.1784

如果说上一篇 GAN 论文是开始出现 GAN 这个让人觉得眼前一亮的模型框架，这篇 cGAN 就是当前 GAN 模型技术变得这么热门的重要因素之一，事实上 GAN 开始是一个无监督模型，生成器需要的仅仅是随机噪声，但是效果并没有那么好，在 14 年提出，到 16 年之前，其实这方面的研究并不多，真正开始一大堆相关论文发表出来，第一个因素就是 cGAN，第二个因素是等会介绍的 DCGAN；

cGAN 其实是将 GAN 又拉回到**监督学习**领域，它在生成器部分添加了**类别标签这个输入**，通过这个改进，缓和了 GAN 的一大问题–训练不稳定，而这种思想，引入先验知识的做法，在如今大多数非常有名的 GAN 中都采用这种做法，后面介绍的生成图片的 BigGAN，或者是图片转换的 Pix2Pix，都是这种思想，可以说 cGAN 的提出非常关键。

Generative Adversarial Networks实际上是对D和G解决以下极小化极大的二元博弈问题：

![gan](https://user-images.githubusercontent.com/83259959/168706412-c6184bbf-0fdd-41b5-beeb-6e872202ab04.png)

而在D和G中均加入条件约束y时，实际上就变成了带有条件概率的二元极小化极大问题：

![cgan fig1](https://user-images.githubusercontent.com/83259959/168706496-527166ae-26b4-453f-a532-6a62bb422752.png)

同一般形式的GAN类似，也是先训练判别网络，再训练生成网络，然后再训练判别网络，两个网络交替训练。只是训练判别网络的样本稍有不同，如图4所示，训练判别网络的时候需要这三种样本，分别是：（1）条件和与条件相符的真实图片，期望输出为1；（2）条件和与条件不符的真实图片，期望输出为0；（3）条件和生成网络生成的输出，期望输出为0。

![cgan fig2](https://user-images.githubusercontent.com/83259959/168706632-7d3912c8-c843-489b-a71f-e97435a1dd0a.png)

### 3.DCGAN

论文名称：Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

论文地址：https://arxiv.org/abs/1511.06434

2015年发表的。这是第一次采用 CNN 结构实现 GAN 模型，它介绍如何使用卷积层，并给出一些额外的结构上的指导建议来实现。另外，它还讨论如何可视化 GAN 的特征、隐空间的插值、利用判别器特征训练分类器以及评估结果。下图是 DCGAN 的生成器部分结构示意图

![dcgan](https://user-images.githubusercontent.com/83259959/168706715-94b5d1d7-2e8b-493e-8612-70f4bd0830a6.png)

### 4. Improved Techniques for Training GANs

论文名称：Improved Techniques for Training GANs

论文地址：https://arxiv.org/abs/1606.03498

这篇论文的作者之一是 Ian Goodfellow，它介绍了很多如何构建一个 GAN 结构的建议，它可以帮助你理解 GAN 不稳定性的原因，给出很多稳定训练 DCGANs 的建议，比如特征匹配(feature matching)、最小批次判别(minibatch discrimination)、单边标签平滑(one-sided label smoothing)、虚拟批归一化(virtual batch normalization)等等，利用这些建议来实现 DCGAN 模型是一个很好学习了解 GANs 的做法。

**提出了IS指标**

### 5.WGAN

#### Wasserstein GAN GP

论文：[[Paper\]](https://arxiv.org/abs/1701.07875) [[Code\]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py)

[令人拍案叫绝的Wasserstein GAN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25071913)

#### *Improved Training of Wasserstein GANs*

论文：[[1704.00028\] Improved Training of Wasserstein GANs](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1704.00028)

[生成式对抗网络GAN有哪些最新的发展，可以实际应用到哪些场景中？ - 知乎 (zhihu.com)](

### 6. Pix2Pix

论文名称：Image-to-Image Translation with Conditional Adversarial Networks

论文地址：https://arxiv.org/abs/1611.07004

blog:[(18条消息) Pix2Pix-基于GAN的图像翻译_张雨石的博客-CSDN博客_pix2pix算法](https://blog.csdn.net/stdcoutzyx/article/details/78820728?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1.pc_relevant_default&spm=1001.2101.3001.4242.2&utm_relevant_index=4)

Pix2Pix 的目标是实现图像转换的应用，如下图所示。这个模型在训练时候需要采用成对的训练数据，并对 GAN 模型采用了不同的配置。其中它应用到了 PatchGAN 这个模型，PatchGAN 对图片的一块 70*70 大小的区域进行观察来判断该图片是真是假，而不需要观察整张图片。

此外，生成器部分使用 U-Net 结构，即结合了 ResNet 网络中的 skip connections 技术，编码器和解码器对应层之间有相互连接，它可以实现如下图所示的转换操作，比如语义图转街景，黑白图片上色，素描图变真实照片等。

![image](https://user-images.githubusercontent.com/83259959/168706868-de0a9e93-1999-4a0d-b748-b29896a8a44d.png)

### 7. CycleGAN

论文名称：Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

论文地址：https://arxiv.org/abs/1703.10593

blog:[(18条消息) CycleGAN算法笔记_AI之路的博客-CSDN博客_cycle gan](https://blog.csdn.net/u014380165/article/details/98462928?ops_request_misc=%7B%22request%5Fid%22%3A%22164852375516781683919310%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=164852375516781683919310&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-98462928.142^v5^pc_search_insert_es_download,143^v6^control&utm_term=cyclegan&spm=1018.2226.3001.4187)

上一篇论文 Pix2Pix 的问题就是训练数据必须成对，即需要原图片和对应转换后的图片，而现实就是这种数据非常难寻找，甚至有的不存在这样一对一的转换数据，因此有了 CycleGAN，仅仅需要准备两个领域的数据集即可，比如说普通马的图片和斑马的图片，但不需要一一对应。这篇论文提出了一个非常好的方法–循环一致性(Cycle-Consistency)损失函数，如下图所示的结构：

![image](https://user-images.githubusercontent.com/83259959/168708118-3b44bc80-5ecc-4538-a199-5a6b04f6d3c1.png)

这种结构在接下来图片转换应用的许多 GAN 论文中都有利用到，cycleGAN 可以实现如下图所示的一些应用，普通马和斑马的转换、风格迁移（照片变油画）、冬夏季节变换等等。

### 8. Progressively Growing of GANs

论文名称：Progressive Growing of GANs for Improved Quality, Stability, and Variation

论文地址：https://arxiv.org/abs/1710.10196

blog:[(18条消息) [论文笔记\]：PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION_Axiiiz的博客-CSDN博客](https://blog.csdn.net/sinat_38059712/article/details/104996375?ops_request_misc=%7B%22request%5Fid%22%3A%22164855538116782246482342%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=164855538116782246482342&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-5-104996375.142^v5^pc_search_insert_es_download,143^v6^control&utm_term=PROGRESSIVE+GROWING+OF+GANS+FOR+IMPROVED+QUALITY%2C+STABILITY%2C+AND+VARIATION&spm=1018.2226.3001.4187)

这篇论文必读的原因是因为它取得非常好的结果以及对于 GAN 问题的创造性方法。它利用一个多尺度结构，从 `4*4` 到 `8*8` 一直提升到 `1024*1024` 的分辨率，如下图所示的结构，这篇论文提出了一些如何解决由于目标图片尺寸导致的不稳定问题。

![image](https://user-images.githubusercontent.com/83259959/168706919-98c7089c-cbe7-4e4a-bedc-513d6f7486d5.png)

### 9. BigGAN

论文地址：Large Scale GAN Training for High Fidelity Natural Image Synthesis

论文地址：https://arxiv.org/abs/1809.11096

BigGAN 应该是当前 ImageNet 上图片生成最好的模型了，它的生成结果如下图所示，非常的逼真，但这篇论文比较难在本地电脑上进行复现，它同时结合了很多结构和技术，包括自注意机制(Self-Attention)、谱归一化(Spectral Normalization)等，这些在论文都有很好的介绍和说明。

### 10. StyleGAN

论文地址：A Style-Based Generator Architecture for Generative Adversarial Networks

论文地址：https://arxiv.org/abs/1812.04948

StyleGAN 借鉴了如 Adaptive Instance Normalization (AdaIN)的自然风格转换技术，来控制隐空间变量 `z` 。其网络结构如下图所示，它在生产模型中结合了一个映射网络以及 AdaIN 条件分布的做法，并不容易复现，但这篇论文依然值得一读，包含了很多有趣的想法。

![image](https://user-images.githubusercontent.com/83259959/168707110-790edee7-db03-4abe-a294-e7a5e2f3f159.png)

### 11.ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-Identification in Multispectral Dataset

提出了一个用于跨模态彩色热人员重新识别 (ReID) 的 ThermalGAN 框架。 我们使用一堆生成对抗网络 (GAN) 将单色探针图像转换为多模态热探针集。 我们使用热直方图和特征描述符作为热特征。



### 12. UNIT

Paper：(NeurIPS 2017) Unsupervised Image-to-Image Translation Networks

Link：https://proceedings.neurips.cc/paper/2017/file/dc6a6489640ca02b0d42dabeb8e46bb7-Paper.pdf

Abstract:

Unsupervised image-to-image translation aims at learning a joint distribution of images in different domains by using images from the marginal distributions in individual domains. Since there exists an infinite set of joint distributions that can arrive the given marginal distributions, one could infer nothing about the joint distribution from the marginal distributions without additional assumptions. To address the problem, we make a shared-latent space assumption and propose an unsupervised image-to-image translation framework based on Coupled GANs. We compare the proposed framework with competing approaches and present high quality image translation results on various challenging unsupervised image translation tasks, including street scene image translation, animal image translation, and face image translation. We also apply the proposed framework to domain adaptation and achieve state-of-the-art performance on benchmark datasets. Code and additional results are available in https://github.com/mingyuliutw/unit

Checkpoint:
https://drive.google.com/drive/folders/1UJ5nOuc4jYwEot1uEQQgnURMXi4Q-QdL?usp=sharing

