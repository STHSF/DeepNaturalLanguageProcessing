# Word Segmentation（中文分词系统）
基于bidirectional_lstm的中文分词，实际上属于序列标注的问题。本文使用的就是字标注法做中文分词。

# 语料库预处理
训练语料库的预处理主要是将不同格式的训练语料处理成RNN的统一输入格式。

本文选取的标注好的[语料库](http://kexue.fm/usr/uploads/2016/10/1372394625.zip)由很多短句组成,下面摘录了部分训练语料数据。
```
“/s  人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e  ，/s  而/s  血/s  与/s  火/s  的/s  战/b  争/e  更/s  是/s  不/b  可/m  多/m  得/e  的/s  教/b  科/m  书/e  ，/s  她/s  确/b  实/e  是/s  名/b  副/m  其/m  实/e  的/s  ‘/s  我/s  的/s  大/b  学/e  ’/s  。/s  
“/s  心/s  静/s  渐/s  知/s  春/s  似/s  海/s  ，/s  花/s  深/s  每/s  觉/s  影/s  生/s  香/s  。/s  
“/s  吃/s  屎/s  的/s  东/b  西/e  ，/s  连/s  一/s  捆/s  麦/s  也/s  铡/s  不/s  动/s  呀/s  ？/s  
他/s  “/s  严/b  格/m  要/m  求/e  自/b  己/e  ，/s  从/s  一/b  个/e  科/b  举/e  出/b  身/e  的/s  进/b  士/e  成/b  为/e  一/b  个/e  伟/b  大/e  的/s  民/b  主/m  主/m  义/e  者/s  ，/s  进/b  而/e  成/b  为/e  一/s  位/s  杰/b  出/e  的/s  党/b  外/e  共/b  产/m  主/m  义/e  战/b  士/e  ，/s  献/b  身/e  于/s  崇/b  高/e  的/s  共/b  产/m  主/m  义/e  事/b  业/e  。/s  
“/s  征/s  而/s  未/s  用/s  的/s  耕/b  地/e  和/s  有/s  收/b  益/e  的/s  土/b  地/e  ，/s  不/b  准/e  荒/b  芜/e  。/s  
“/s  这/s  首/b  先/e  是/s  个/s  民/b  族/e  问/b  题/e  ，/s  民/b  族/e  的/s  感/b  情/e  问/b  题/e  。/s  
```
_针对该语料库预处理的关键步骤：_

- **step1、** 脏数据清洗，剔除一些不规范的字符串等内容。比如(每句开头的单双引号)。

- **step2、** 将所有句子和段落连接成整体，然后按照里面的标点符号重新切句。

- **step3、** 将每一句中的字和标签分开存储，比如分成[[word1, word2,...., wordi],[word1, word2,...., wordj],...,[word1, word2,...., wordk]]和[[tag1, tag2,....,tagi], [tag1, tag2,....,tagj],....,[tag1, tag2,....,tagk]]的形式,list中的每个list代表一句话中的词。

- **step4、** 统计所有的字的个数和标签类别的个数，并为每一个字和标签编号，构建words和tags都转为{数值-->id}的映射,包括[word_to_id, id_to_word, tag_to_id, id_to_tag]。

- **step5、** padding的过程。将step3生成的句子列表中的每一句padding成固定长度的字列表，具体做法是对于长度小于固定长度的句子使用0填充到固定长度，长度大于固定长度的句子则将超过的部分切除。h最后变成[[word1, word2,...., wordn],[word1, word2,...., wordn],...,[word1, word2,...., wordn]]和[[tag1, tag2,....,tagn], [tag1, tag2,....,tagn],....,[tag1, tag2,....,tagn]],其中n为固定长度。

- **step6、** 隐藏状态的转移概率矩阵计算。这是为后面使用 viterbi 进行decoding准备的，具体解释参见viterbi算法。

以上数据处理好后可以分单元将数据分别保存为pickle文件，后面使用的时候直接load就可以了。


# batch数据准备
模型的训练过程中一般采用mini-batch的方式feeding数据。所以需要对源数据进行Batch_generator, 即将train_data, valid_data, test_data按照batch_size且分开。
处理后的数据的shape为[batch_size, time_steps]


# 构建Bidirectional_RNN单元
具体Bidirectional_RNN的理解参考[这篇文章](https://sthsf.github.io/2017/08/31/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86-bidirectional-rnn/),



**模型的训练结果：**
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
1、简单来说，机器学习就是根据样本（即数据）学习得到一个模型，再根据这个模型预测的一种方法。
2、一起恶性交通事故
3、结婚的和尚未结婚的
4、在北京大学生活区喝进口红酒
5、阿美首脑会议将讨论巴以和平等问题
6、让我们以爱心和平等来对待动物

```
分词结果：
```
1、简单 / 来 / 说 / ， / 机器 / 学习 / 就 / 是 / 根据 / 样本 / （ / 即数 / 据 / ） / 学习 / 得到 / 一个 / 模型 / ， / 再 / 根据 / 这个 / 模型 / 预测 / 的 / 一种 / 方法 / 。 / 
2、一 / 起 / 恶性 / 交通 / 事故 / 
3、结婚 / 的 / 和 / 尚未 / 结婚 / 的 / 
4、在 / 北京 / 大学生 / 活区 / 喝 / 进口 / 红酒 /
5、阿美 / 首脑 / 会议 / 将 / 讨论 / 巴以 / 和平 / 等 / 问题 /
6、让 / 我们 / 以 / 爱心 / 和平 / 等 / 来 / 对待 / 动物 / 
```


## 注意点




#### 参考文献
[tensorflow中mask](http://blog.csdn.net/appleml/article/details/56675152)
[Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
[PKU](http://sighan.cs.uchicago.edu/bakeoff2005/data/pku_spec.pdf)
[TF使用例子-LSTM实现序列标注](http://www.jianshu.com/p/4cfcce68fc3b)
[Tensorflow下构建LSTM模型进行序列化标注](http://www.deepnlp.org/blog/tensorflow-lstm-pos/)
[使用深度学习进行中文自然语言处理之序列标注](http://www.jianshu.com/p/7e233ef57cb6)
[使用RNN解决NLP中序列标注问题的通用优化思路](http://blog.csdn.net/malefactor/article/details/50725480)
[ HMM与序列标注](http://blog.csdn.net/zbc1090549839/article/details/53887031)
[deepnlp](https://github.com/rockingdingo/deepnlp/tree/r0.1.7#segmentation)
[中文分词入门之字标注法4](http://www.52nlp.cn/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%85%A5%E9%97%A8%E4%B9%8B%E5%AD%97%E6%A0%87%E6%B3%A8%E6%B3%954)
[TensorFlow入门（六） 双端 LSTM 实现序列标注（分词）](http://blog.csdn.net/jerr__y/article/details/70471066)
[基于双向LSTM的seq2seq字标注-苏剑林](http://spaces.ac.cn/archives/3924/)
[TF使用例子-LSTM实现序列标注](http://www.jianshu.com/p/4cfcce68fc3b)
[Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)