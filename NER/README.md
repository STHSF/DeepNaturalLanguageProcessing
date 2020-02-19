# 命名实体识别
识别对话里面的人名、地名、组织机构名。属于序列标注问题。
命名实体识别的标注采用的是BIEO的方式，即Begin，Intermediate，End，Other。实际上也就是使用BIEO将目标句中的词按照需求的方式标记，不同的结果取决于对样本数据的标注，一般序列的标注是要符合一定的标注标准的如([PKU数据标注规范](http://sighan.cs.uchicago.edu/bakeoff2005/data/pku_spec.pdf))。
词性标注、分词都属于同一类问题，他们的区别主要是标注的方式不同。

## 命名实体识别中的标签集合
标注方式1、
LabelSet = {BA, MA, EA, BO, MO, EO, BP, MP, EP, O}
其中，BA代表这个汉字是地址首字，MA代表这个汉字是地址中间字，EA代表这个汉字是地址的尾字；BO代表这个汉字是机构名的首字，MO代表这个汉字是机构名称的中间字，EO代表这个汉字是机构名的尾字；BP代表这个汉字是人名首字，MP代表这个汉字是人名中间字，EP代表这个汉字是人名尾字，而O代表这个汉字不属于命名实体。
标注方式2、
LabelSet = {NA, SC, CC, SL, LL, SP, PP}
其中 NA = No entity, SC = Start Company, CC = Continue Company, SL = Start Location, CL = Continue Location, SP = Start Person, CP = Continue Person

标注方式3、字符级别的标注：
LabelSet = {O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}
其中，PER代表人名， LOC代表位置， ORG代表组织

一般来说，NER的标注列表为['O' ,'B-MISC', 'I-MISC', 'B-ORG' ,'I-ORG', 'B-PER' ,'I-PER', 'B-LOC' ,'I-LOC']。
其中，一般一共分为四大类：PER（人名），LOC（位置），ORG（组织）以及MISC(杂项)，而且B表示开始，I表示中间，O表示单字词。




# 參考文獻
[NER](https://github.com/shiyybua/NER/blob/master/utils.py)
[基础却不简单，命名实体识别的难点与现状](https://zhuanlan.zhihu.com/p/26782938)
[python+HMM之维特比解码](http://blog.csdn.net/jerr__y/article/details/73838805)
[Tagging Problems, and Hidden Markov Models](http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf)
[]()
[]()
[]()
[]()