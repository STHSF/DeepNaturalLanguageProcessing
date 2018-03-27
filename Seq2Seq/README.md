# 
本文模仿的是使用seq2seq模型来训练模型，复制原来的序列
### 训练结果：
```
batch 0
  minibatch loss: 2.11096167564
  sample 1:
    input     > [5 2 8 4 6 6 3 2]
    predicted > [7 0 0 0 0 0 0 0 0]
  sample 2:
    input     > [9 5 4 2 6 4 5 6]
    predicted > [2 2 0 0 0 0 0 0 0]
  sample 3:
    input     > [4 8 8 9 2 9 3 0]
    predicted > [0 0 0 0 0 0 0 0 0]
batch 1000
  minibatch loss: 0.0359240211546
  sample 1:
    input     > [6 5 6 4 8 6 0 0]
    predicted > [6 5 6 4 8 6 1 0 0]
  sample 2:
    input     > [5 4 2 9 3 0 0 0]
    predicted > [5 4 2 9 3 1 0 0 0]
  sample 3:
    input     > [4 2 9 0 0 0 0 0]
    predicted > [4 2 9 1 0 0 0 0 0]
batch 2000
  minibatch loss: 0.0119506847113
  sample 1:
    input     > [9 4 7 4 2 0 0 0]
    predicted > [9 4 7 4 2 1 0 0 0]
  sample 2:
    input     > [7 2 6 6 6 9 4 0]
    predicted > [7 2 6 6 6 9 4 1 0]
  sample 3:
    input     > [4 5 4 6 2 0 0 0]
    predicted > [4 5 4 6 2 1 0 0 0]
batch 3000
  minibatch loss: 0.00151053641457
  sample 1:
    input     > [8 8 3 3 6 7 7 0]
    predicted > [8 8 3 3 6 7 7 1 0]
  sample 2:
    input     > [2 8 5 7 0 0 0 0]
    predicted > [2 8 5 7 1 0 0 0 0]
  sample 3:
    input     > [4 8 9 9 6 9 8 3]
    predicted > [4 8 9 9 6 9 8 3 1]
```

## reference
[深度学习要多深，才能读懂人话？阿里小蜜前沿探索](https://baijiahao.baidu.com/s?id=1570986733987329&wfr=spider&for=pc)
[颠覆传统的电商智能助理-阿里小蜜技术揭秘](http://www.infoq.com/cn/articles/electricity-supplier-intelligent-assistant/)
[揭秘阿里小蜜：基于检索模型和生成模型相结合的聊天引擎](http://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/78849003)
[阿里小蜜技术学习笔记--知识点整理](http://blog.csdn.net/l18930738887/article/details/55667427)