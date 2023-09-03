
# 改进后的PSPNET 分别接入了CBAM ECA SE 模块。
此次未在主干网络中修改，如有需要可自主修改，但得重新训练权重，无法使用预训练权重进行训练，可以简单理解为拿VOC再跑一个出来，但对硬件资源要求较大。


# 网络改进部分如图所示 
![image](https://github.com/vitant-lang/-improved-PSPNET/assets/75409802/a28fb714-e899-485a-ab46-22c83e28a57e)

尝试过在PPM模块加入，发现导致计算量极大，并且提升的指标并不是很高。

显存暴毙问题 如果一开始无法训练，应该是batch size的问题，如果训练一会爆了，个人经验是调整numworker的值，（在文件里CTRL加F搜）。


