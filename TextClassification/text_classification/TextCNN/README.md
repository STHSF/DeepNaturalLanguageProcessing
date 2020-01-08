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
### CNN
```shell script
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  15.62%, Val Loss:    2.3, Val Acc:  10.00%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.64, Train Acc:  81.25%, Val Loss:   0.98, Val Acc:  70.06%, Time: 0:00:03 *
Iter:    200, Train Loss:    0.3, Train Acc:  90.62%, Val Loss:   0.55, Val Acc:  82.54%, Time: 0:00:05 *
Iter:    300, Train Loss:   0.11, Train Acc:  96.88%, Val Loss:   0.45, Val Acc:  85.48%, Time: 0:00:07 *
Iter:    400, Train Loss:   0.35, Train Acc:  89.06%, Val Loss:   0.36, Val Acc:  88.36%, Time: 0:00:09 *
Iter:    500, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:    0.3, Val Acc:  90.72%, Time: 0:00:11 *
Iter:    600, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.27, Val Acc:  90.94%, Time: 0:00:13 *
Iter:    700, Train Loss:   0.31, Train Acc:  87.50%, Val Loss:   0.24, Val Acc:  92.12%, Time: 0:00:15 *
Epoch: 2
Iter:    800, Train Loss:  0.081, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.64%, Time: 0:00:17 *
Iter:    900, Train Loss:   0.39, Train Acc:  84.38%, Val Loss:   0.27, Val Acc:  91.08%, Time: 0:00:19
Iter:   1000, Train Loss:   0.26, Train Acc:  95.31%, Val Loss:   0.24, Val Acc:  91.56%, Time: 0:00:21
Iter:   1100, Train Loss:  0.081, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  93.42%, Time: 0:00:23
Iter:   1200, Train Loss:    0.2, Train Acc:  95.31%, Val Loss:   0.26, Val Acc:  91.50%, Time: 0:00:25
Iter:   1300, Train Loss:   0.28, Train Acc:  92.19%, Val Loss:   0.27, Val Acc:  91.44%, Time: 0:00:27
Iter:   1400, Train Loss:  0.086, Train Acc:  96.88%, Val Loss:   0.17, Val Acc:  94.72%, Time: 0:00:29 *
Iter:   1500, Train Loss:  0.044, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  92.50%, Time: 0:00:31
Epoch: 3
Iter:   1600, Train Loss:   0.05, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.58%, Time: 0:00:33
Iter:   1700, Train Loss:  0.019, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  92.16%, Time: 0:00:35
Iter:   1800, Train Loss:  0.015, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.16%, Time: 0:00:37
Iter:   1900, Train Loss:  0.026, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  93.28%, Time: 0:00:39
Iter:   2000, Train Loss:   0.11, Train Acc:  95.31%, Val Loss:   0.19, Val Acc:  93.80%, Time: 0:00:41
Iter:   2100, Train Loss:  0.089, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  92.46%, Time: 0:00:43
Iter:   2200, Train Loss:  0.075, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  93.42%, Time: 0:00:45
Iter:   2300, Train Loss:  0.029, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.46%, Time: 0:00:48
Epoch: 4
Iter:   2400, Train Loss:  0.048, Train Acc:  96.88%, Val Loss:   0.22, Val Acc:  93.14%, Time: 0:00:50
No optimization for a long time, auto-stopping...
```