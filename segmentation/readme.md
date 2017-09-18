# segmentation（中文分词系统）
基于bidirectional_lstm的中文分词，实际上属于序列标注的问题。本文使用的就是字标注法做中文分词。

模型的训练结果：
```
EPOCH 1， lr=0.0001
	training acc=0.741334, cost=0.589564;  valid acc= 0.778319, cost=0.468106 
	training acc=0.792995, cost=0.443801;  valid acc= 0.807903, cost=0.423901 
	training acc=0.830177, cost=0.390488;  valid acc= 0.854199, cost=0.354184 
	training acc=0.868506, cost=0.328745;  valid acc= 0.879461, cost=0.307371 
	training acc=0.884908, cost=0.296304;  valid acc= 0.890496, cost=0.284209 
	training 205780, acc=0.823662, cost=0.409642 
Epoch training 205780, acc=0.823662, cost=0.409642, speed=628.616 s/epoch
EPOCH 2， lr=0.0001
	training acc=0.893218, cost=0.276941;  valid acc= 0.896585, cost=0.269476 
	training acc=0.898869, cost=0.264248;  valid acc= 0.902243, cost=0.256955 
	training acc=0.903743, cost=0.252664;  valid acc= 0.906572, cost=0.24681 
	training acc=0.908525, cost=0.241395;  valid acc= 0.910965, cost=0.236243 
	training acc=0.912821, cost=0.231168;  valid acc= 0.91529, cost=0.225519 
	training 205780, acc=0.903448, cost=0.253248 
Epoch training 205780, acc=0.903448, cost=0.253248, speed=630.298 s/epoch
EPOCH 3， lr=0.0001
	training acc=0.917001, cost=0.220821;  valid acc= 0.919243, cost=0.215995 
	training acc=0.919994, cost=0.21319;  valid acc= 0.922561, cost=0.207766 
	training acc=0.924294, cost=0.202953;  valid acc= 0.925814, cost=0.200045 
	training acc=0.927343, cost=0.196118;  valid acc= 0.928539, cost=0.193171 
	training acc=0.929331, cost=0.190819;  valid acc= 0.930675, cost=0.188498 
the save path is  ckpt/bi-lstm.ckpt-3
	training 205780, acc=0.923587, cost=0.204783 
Epoch training 205780, acc=0.923587, cost=0.204783, speed=666.714 s/epoch
EPOCH 4， lr=0.0001
	training acc=0.931222, cost=0.185732;  valid acc= 0.932741, cost=0.182281 
	training acc=0.935005, cost=0.176828;  valid acc= 0.934603, cost=0.177537 
	training acc=0.93522, cost=0.175972;  valid acc= 0.936216, cost=0.173191 
	training acc=0.937221, cost=0.170854;  valid acc= 0.937602, cost=0.169675 
	training acc=0.938753, cost=0.16684;  valid acc= 0.938713, cost=0.166556 
	training 205780, acc=0.935493, cost=0.175232 
Epoch training 205780, acc=0.935493, cost=0.175232, speed=638.791 s/epoch
EPOCH 5， lr=0.0001
	training acc=0.940126, cost=0.163223;  valid acc= 0.939784, cost=0.163813 
	training acc=0.940796, cost=0.162021;  valid acc= 0.940891, cost=0.161071 
	training acc=0.941845, cost=0.159138;  valid acc= 0.941697, cost=0.158886 
	training acc=0.942423, cost=0.157736;  valid acc= 0.94279, cost=0.156548 
	training acc=0.943627, cost=0.154031;  valid acc= 0.943256, cost=0.154781 
	training 205780, acc=0.94177, cost=0.159211 
Epoch training 205780, acc=0.94177, cost=0.159211, speed=634.48 s/epoch
EPOCH 6， lr=0.0001
	training acc=0.944591, cost=0.15182;  valid acc= 0.943395, cost=0.153735 
	training acc=0.945331, cost=0.150285;  valid acc= 0.94432, cost=0.151445 
	training acc=0.945657, cost=0.148853;  valid acc= 0.945442, cost=0.148962 
	training acc=0.946828, cost=0.146195;  valid acc= 0.946251, cost=0.146829 
	training acc=0.947234, cost=0.144663;  valid acc= 0.94677, cost=0.145869 
the save path is  ckpt/bi-lstm.ckpt-6
	training 205780, acc=0.945926, cost=0.148367 
Epoch training 205780, acc=0.945926, cost=0.148367, speed=659.901 s/epoch
**TEST RESULT:
**Test 64307, acc=0.946648, cost=0.146562
```

### 分词效果：
测试句：
```
简单来说，机器学习就是根据样本（即数据）学习得到一个模型，再根据这个模型预测的一种方法。
```
分词结果：
```
简单 / 来 / 说 / ， / 机器 / 学习 / 就 / 是 / 根据 / 样本 / （ / 即数 / 据 / ） / 学习 / 得到 / 一个 / 模型 / ， / 再 / 根据 / 这个 / 模型 / 预测 / 的 / 一种 / 方法 / 。 / 
```


## 注意点




#### 参考文献
[基于双向LSTM的seq2seq字标注-苏剑林](http://spaces.ac.cn/archives/3924/)
[TF使用例子-LSTM实现序列标注](http://www.jianshu.com/p/4cfcce68fc3b)
[Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)