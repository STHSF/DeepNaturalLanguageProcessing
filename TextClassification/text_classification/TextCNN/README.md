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

### CNN  multi-filter
```shell script
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  15.62%, Val Loss:    2.3, Val Acc:  10.00%, Time: 0:00:02 *
Iter:    100, Train Loss:    1.1, Train Acc:  79.69%, Val Loss:    1.4, Val Acc:  67.06%, Time: 0:00:09 *
Iter:    200, Train Loss:   0.28, Train Acc:  90.62%, Val Loss:    0.7, Val Acc:  79.06%, Time: 0:00:16 *
Iter:    300, Train Loss:   0.49, Train Acc:  84.38%, Val Loss:   0.57, Val Acc:  81.40%, Time: 0:00:23 *
Iter:    400, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.49, Val Acc:  85.20%, Time: 0:00:30 *
Iter:    500, Train Loss:   0.32, Train Acc:  85.94%, Val Loss:   0.39, Val Acc:  86.60%, Time: 0:00:37 *
Iter:    600, Train Loss:   0.35, Train Acc:  89.06%, Val Loss:    0.3, Val Acc:  90.40%, Time: 0:00:44 *
Iter:    700, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.28, Val Acc:  91.06%, Time: 0:00:51 *
Epoch: 2
Iter:    800, Train Loss:    0.3, Train Acc:  92.19%, Val Loss:   0.26, Val Acc:  91.56%, Time: 0:00:58 *
Iter:    900, Train Loss:   0.21, Train Acc:  92.19%, Val Loss:    0.2, Val Acc:  94.30%, Time: 0:01:05 *
Iter:   1000, Train Loss:   0.14, Train Acc:  96.88%, Val Loss:   0.28, Val Acc:  90.94%, Time: 0:01:12
Iter:   1100, Train Loss:  0.056, Train Acc:  98.44%, Val Loss:   0.22, Val Acc:  93.86%, Time: 0:01:19
Iter:   1200, Train Loss:   0.12, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  93.58%, Time: 0:01:26
Iter:   1300, Train Loss:   0.26, Train Acc:  90.62%, Val Loss:    0.2, Val Acc:  94.54%, Time: 0:01:33 *
Iter:   1400, Train Loss:  0.083, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  93.64%, Time: 0:01:40
Iter:   1500, Train Loss:  0.077, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  95.08%, Time: 0:01:47 *
Epoch: 3
Iter:   1600, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.16, Val Acc:  95.72%, Time: 0:01:54 *
Iter:   1700, Train Loss:  0.092, Train Acc:  96.88%, Val Loss:   0.21, Val Acc:  93.58%, Time: 0:02:01
Iter:   1800, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:   0.17, Val Acc:  95.60%, Time: 0:02:08
Iter:   1900, Train Loss:  0.062, Train Acc:  96.88%, Val Loss:   0.19, Val Acc:  95.06%, Time: 0:02:15
Iter:   2000, Train Loss:  0.078, Train Acc:  96.88%, Val Loss:   0.19, Val Acc:  95.00%, Time: 0:02:22
Iter:   2100, Train Loss:   0.17, Train Acc:  93.75%, Val Loss:   0.23, Val Acc:  93.06%, Time: 0:02:29
Iter:   2200, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.25, Val Acc:  92.48%, Time: 0:02:36
Iter:   2300, Train Loss:  0.068, Train Acc:  95.31%, Val Loss:   0.19, Val Acc:  94.88%, Time: 0:02:43
Epoch: 4
Iter:   2400, Train Loss:  0.012, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  94.56%, Time: 0:02:50
Iter:   2500, Train Loss:  0.049, Train Acc:  98.44%, Val Loss:   0.19, Val Acc:  94.76%, Time: 0:02:57
Iter:   2600, Train Loss:  0.033, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  95.60%, Time: 0:03:04
No optimization for a long time, auto-stopping...
```
### 第二次运行结果：
```shell script
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:   9.38%, Val Loss:    2.3, Val Acc:  10.00%, Time: 0:00:02 *
Iter:    100, Train Loss:    1.2, Train Acc:  73.44%, Val Loss:    1.4, Val Acc:  62.56%, Time: 0:00:09 *
Iter:    200, Train Loss:   0.37, Train Acc:  90.62%, Val Loss:   0.73, Val Acc:  80.16%, Time: 0:00:16 *
Iter:    300, Train Loss:   0.25, Train Acc:  93.75%, Val Loss:   0.49, Val Acc:  84.52%, Time: 0:00:23 *
Iter:    400, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.44, Val Acc:  85.48%, Time: 0:00:30 *
Iter:    500, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.34, Val Acc:  89.70%, Time: 0:00:37 *
Iter:    600, Train Loss:   0.13, Train Acc:  98.44%, Val Loss:   0.32, Val Acc:  89.94%, Time: 0:00:44 *
Iter:    700, Train Loss:   0.19, Train Acc:  92.19%, Val Loss:   0.27, Val Acc:  91.70%, Time: 0:00:51 *
Epoch: 2
Iter:    800, Train Loss:   0.21, Train Acc:  92.19%, Val Loss:   0.27, Val Acc:  91.40%, Time: 0:00:58
Iter:    900, Train Loss:   0.25, Train Acc:  92.19%, Val Loss:   0.24, Val Acc:  92.58%, Time: 0:01:05 *
Iter:   1000, Train Loss:  0.074, Train Acc:  96.88%, Val Loss:   0.25, Val Acc:  92.22%, Time: 0:01:12
Iter:   1100, Train Loss:  0.063, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  92.38%, Time: 0:01:19
Iter:   1200, Train Loss:  0.087, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  91.96%, Time: 0:01:26
Iter:   1300, Train Loss:  0.075, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  93.48%, Time: 0:01:33 *
Iter:   1400, Train Loss:   0.17, Train Acc:  93.75%, Val Loss:    0.2, Val Acc:  93.60%, Time: 0:01:40 *
Iter:   1500, Train Loss:  0.034, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  93.34%, Time: 0:01:47
Epoch: 3
Iter:   1600, Train Loss:   0.02, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  93.98%, Time: 0:01:54 *
Iter:   1700, Train Loss:   0.11, Train Acc:  95.31%, Val Loss:    0.2, Val Acc:  93.82%, Time: 0:02:01
Iter:   1800, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.19, Val Acc:  94.22%, Time: 0:02:08 *
Iter:   1900, Train Loss:   0.03, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.72%, Time: 0:02:15 *
Iter:   2000, Train Loss:    0.1, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.52%, Time: 0:02:22
Iter:   2100, Train Loss:  0.034, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  94.90%, Time: 0:02:29 *
Iter:   2200, Train Loss:   0.18, Train Acc:  92.19%, Val Loss:    0.2, Val Acc:  94.06%, Time: 0:02:36
Iter:   2300, Train Loss:  0.069, Train Acc:  98.44%, Val Loss:   0.17, Val Acc:  95.18%, Time: 0:02:43 *
Epoch: 4
Iter:   2400, Train Loss:  0.039, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.74%, Time: 0:02:50
Iter:   2500, Train Loss:  0.081, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  94.28%, Time: 0:02:57
Iter:   2600, Train Loss:  0.052, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.42%, Time: 0:03:05 *
Iter:   2700, Train Loss:  0.034, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  92.46%, Time: 0:03:12
Iter:   2800, Train Loss:  0.017, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  93.96%, Time: 0:03:19
Iter:   2900, Train Loss:   0.12, Train Acc:  93.75%, Val Loss:   0.16, Val Acc:  95.50%, Time: 0:03:26 *
Iter:   3000, Train Loss:   0.03, Train Acc: 100.00%, Val Loss:   0.18, Val Acc:  95.00%, Time: 0:03:33
Iter:   3100, Train Loss:  0.024, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.14%, Time: 0:03:40 *
Epoch: 5
Iter:   3200, Train Loss:  0.048, Train Acc:  96.88%, Val Loss:   0.18, Val Acc:  95.04%, Time: 0:03:47
Iter:   3300, Train Loss:  0.037, Train Acc:  98.44%, Val Loss:    0.2, Val Acc:  93.54%, Time: 0:03:54
Iter:   3400, Train Loss: 0.0079, Train Acc: 100.00%, Val Loss:   0.15, Val Acc:  95.76%, Time: 0:04:01
Iter:   3500, Train Loss:  0.032, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  95.98%, Time: 0:04:08
Iter:   3600, Train Loss:  0.065, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.44%, Time: 0:04:15
Iter:   3700, Train Loss:  0.027, Train Acc:  98.44%, Val Loss:   0.17, Val Acc:  95.22%, Time: 0:04:22
Iter:   3800, Train Loss: 0.0082, Train Acc: 100.00%, Val Loss:   0.15, Val Acc:  95.60%, Time: 0:04:29
Iter:   3900, Train Loss:  0.034, Train Acc:  98.44%, Val Loss:   0.15, Val Acc:  95.54%, Time: 0:04:36
Epoch: 6
Iter:   4000, Train Loss:  0.022, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.70%, Time: 0:04:43
Iter:   4100, Train Loss:  0.055, Train Acc:  98.44%, Val Loss:   0.13, Val Acc:  96.48%, Time: 0:04:50 *
Iter:   4200, Train Loss:  0.039, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.64%, Time: 0:04:57
Iter:   4300, Train Loss:  0.012, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  96.22%, Time: 0:05:04
Iter:   4400, Train Loss: 0.0049, Train Acc: 100.00%, Val Loss:   0.15, Val Acc:  96.06%, Time: 0:05:11
Iter:   4500, Train Loss:  0.042, Train Acc:  96.88%, Val Loss:   0.14, Val Acc:  96.44%, Time: 0:05:18
Iter:   4600, Train Loss:  0.056, Train Acc:  98.44%, Val Loss:   0.14, Val Acc:  96.38%, Time: 0:05:25
Epoch: 7
Iter:   4700, Train Loss:  0.023, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.46%, Time: 0:05:32
Iter:   4800, Train Loss: 0.0025, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.26%, Time: 0:05:39
Iter:   4900, Train Loss:  0.012, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  95.10%, Time: 0:05:46
Iter:   5000, Train Loss: 0.0034, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  94.04%, Time: 0:05:53
Iter:   5100, Train Loss:  0.097, Train Acc:  98.44%, Val Loss:   0.17, Val Acc:  95.22%, Time: 0:06:00
No optimization for a long time, auto-stopping...
```