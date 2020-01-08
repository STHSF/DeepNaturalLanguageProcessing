# 文本分类

## 写在前面

## 数据集
程序中使用的数据集是新闻文本，总共有十个类别的新闻文本。然后通过模型训练，将测试数据按照十个类别分开。

## 数据预处理
1、本程序处理的最小单元是中文字符，暂时没有涉及到分词。所以需要对训练集做一些预处理。

首先根据文本的存储格式，将标签和正文分别提取出来，处理过程中注意中文的编码。

第二步就是构建词汇集，本文使用的是有监督的训练，所以事先剔除了那些不常用的字，选取了5000个词汇。

然后就是将中文字符和category分别处理成{word: id}和{category: id}的形式。


2、模型的输入数据准备

1）由于模型接受固定长度的输入，所以需要把每篇文章padding成固定长度。

2）batch数据准备，模型在feeding的过程中每次feed进入的是一个batch的数据，所以先要把训练集整理成bath的数据


## 结果
### RNN

### BiRNN
```shell
0
Iter:      0, Train Loss:    2.3, Train Acc:   9.38%, Val Loss:    2.3, Val Acc:   9.64%, Time: 0:00:08 *
Iter:    100, Train Loss:    1.3, Train Acc:  46.88%, Val Loss:    1.5, Val Acc:  40.88%, Time: 0:01:12 *
Iter:    200, Train Loss:   0.81, Train Acc:  71.88%, Val Loss:    1.1, Val Acc:  65.94%, Time: 0:02:15 *
Iter:    300, Train Loss:   0.59, Train Acc:  84.38%, Val Loss:   0.79, Val Acc:  72.14%, Time: 0:03:19 *
Epoch: 2
Iter:    400, Train Loss:   0.38, Train Acc:  88.28%, Val Loss:   0.67, Val Acc:  78.96%, Time: 0:04:22 *
Iter:    500, Train Loss:   0.41, Train Acc:  89.84%, Val Loss:   0.53, Val Acc:  82.24%, Time: 0:05:25 *
Iter:    600, Train Loss:   0.39, Train Acc:  88.28%, Val Loss:   0.53, Val Acc:  82.38%, Time: 0:06:28 *
Iter:    700, Train Loss:    0.3, Train Acc:  90.62%, Val Loss:   0.53, Val Acc:  82.20%, Time: 0:07:31
Epoch: 3
Iter:    800, Train Loss:   0.18, Train Acc:  93.75%, Val Loss:    0.5, Val Acc:  85.24%, Time: 0:08:35 *
Iter:    900, Train Loss:   0.31, Train Acc:  92.19%, Val Loss:   0.52, Val Acc:  84.52%, Time: 0:09:38
Iter:   1000, Train Loss:   0.38, Train Acc:  89.06%, Val Loss:   0.43, Val Acc:  86.22%, Time: 0:10:41 *
Iter:   1100, Train Loss:    0.3, Train Acc:  92.19%, Val Loss:   0.37, Val Acc:  88.36%, Time: 0:11:44 *
Epoch: 4
Iter:   1200, Train Loss:   0.11, Train Acc:  96.09%, Val Loss:   0.37, Val Acc:  88.68%, Time: 0:12:47 *
Iter:   1300, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.34, Val Acc:  89.84%, Time: 0:13:50 *
Iter:   1400, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.36, Val Acc:  89.04%, Time: 0:14:54
Iter:   1500, Train Loss:  0.094, Train Acc:  97.66%, Val Loss:   0.35, Val Acc:  89.58%, Time: 0:15:57
Epoch: 5
Iter:   1600, Train Loss:   0.24, Train Acc:  92.97%, Val Loss:   0.38, Val Acc:  89.46%, Time: 0:17:00
Iter:   1700, Train Loss:   0.12, Train Acc:  95.31%, Val Loss:   0.37, Val Acc:  88.94%, Time: 0:18:03
Iter:   1800, Train Loss:  0.082, Train Acc:  96.88%, Val Loss:    0.3, Val Acc:  91.62%, Time: 0:19:06 *
Iter:   1900, Train Loss:  0.093, Train Acc:  97.66%, Val Loss:   0.32, Val Acc:  90.54%, Time: 0:20:09
Epoch: 6
Iter:   2000, Train Loss:  0.051, Train Acc:  98.44%, Val Loss:    0.3, Val Acc:  91.50%, Time: 0:21:13
Iter:   2100, Train Loss:   0.16, Train Acc:  96.09%, Val Loss:   0.32, Val Acc:  91.20%, Time: 0:22:16
Iter:   2200, Train Loss:  0.052, Train Acc:  98.44%, Val Loss:   0.29, Val Acc:  92.12%, Time: 0:23:19 *
Iter:   2300, Train Loss:  0.099, Train Acc:  96.09%, Val Loss:   0.36, Val Acc:  90.52%, Time: 0:24:22
Epoch: 7
Iter:   2400, Train Loss:   0.07, Train Acc:  96.09%, Val Loss:   0.35, Val Acc:  90.60%, Time: 0:25:25
Iter:   2500, Train Loss:   0.14, Train Acc:  93.75%, Val Loss:   0.32, Val Acc:  91.34%, Time: 0:26:29
Iter:   2600, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:   0.38, Val Acc:  90.56%, Time: 0:27:32
Iter:   2700, Train Loss:  0.068, Train Acc:  97.66%, Val Loss:   0.35, Val Acc:  91.30%, Time: 0:28:35
Epoch: 8
Iter:   2800, Train Loss:  0.074, Train Acc:  98.44%, Val Loss:   0.33, Val Acc:  91.46%, Time: 0:29:38
Iter:   2900, Train Loss:  0.047, Train Acc:  98.44%, Val Loss:   0.34, Val Acc:  92.16%, Time: 0:30:41 *
Iter:   3000, Train Loss:  0.044, Train Acc:  99.22%, Val Loss:   0.33, Val Acc:  92.12%, Time: 0:31:44
Iter:   3100, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:   0.33, Val Acc:  91.70%, Time: 0:32:47
Epoch: 9
Iter:   3200, Train Loss:  0.056, Train Acc:  98.44%, Val Loss:   0.28, Val Acc:  93.48%, Time: 0:33:51 *
Iter:   3300, Train Loss:  0.061, Train Acc:  99.22%, Val Loss:   0.34, Val Acc:  91.76%, Time: 0:34:54
Iter:   3400, Train Loss:  0.033, Train Acc:  98.44%, Val Loss:   0.33, Val Acc:  91.66%, Time: 0:35:57
Iter:   3500, Train Loss:  0.037, Train Acc:  98.44%, Val Loss:   0.37, Val Acc:  91.06%, Time: 0:37:00
Epoch: 10
Iter:   3600, Train Loss:   0.11, Train Acc:  97.66%, Val Loss:    0.3, Val Acc:  93.26%, Time: 0:38:03
Iter:   3700, Train Loss:  0.025, Train Acc:  98.44%, Val Loss:   0.34, Val Acc:  92.12%, Time: 0:39:06
Iter:   3800, Train Loss:  0.014, Train Acc:  99.22%, Val Loss:   0.32, Val Acc:  92.52%, Time: 0:40:09
Iter:   3900, Train Loss:  0.023, Train Acc:  99.22%, Val Loss:   0.33, Val Acc:  92.26%, Time: 0:41:13
```
