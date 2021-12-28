@[TOC](paddle 使用预训练模型做NER命名实体识别任务)

# 前言
**paddlepaddle**是百度的一个深度学习框架,该框架的生态环境中，有一个**paddleNLP**开源子项目，该项目提供了当前大部分NLP预训练模型训练好的模型参数，我们可以使用paddleNLP提供的预训练模型来做NER命名实体识别任务.**本文，主要介绍本人近期，做的关于NER命名实体识别的一些工作**。

# 一：paddle NRE
## （1）开源链接
开源链接：https://gitee.com/lingcb/paddle-ner

## （2）预训练模型
该项目，使用的预训练模型为：**bert-wwm-ext-chinese，，bert-base-multilingual-uncased**
若要选择其他模型，可以查询链接**https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html#transformer**

## （3）数据集
使用了三个数据集：**msra_ner，，peoples_daily_ner，，cluener**
msra_ner，，peoples_daily_ner ，，可以直接使用**paddlenlp.datasets.load_dataset**导入


cluener 数据集，开源链接为 **https://github.com/CLUEbenchmark/CLUENER2020**
同时 **src/cluener_dataset.py** 是处理cluener 数据集的代码

## （4）模型的序列标注
数据集的标注格式是 **BIO方式**

```bash
cluener_label_list = ['B-address', 'I-address', 'B-book', 'I-book', 'B-company', 'I-company', 'B-game', 'I-game', 'B-government', 'I-government', 'B-movie', 'I-movie', 'B-name', 'I-name', 'B-organization', 'I-organization', 'B-position', 'I-position', 'B-scene', 'I-scene', 'O']
msra_label_list = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
peoples_daily_label_list = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
```


## （4）项目结构

```bash
.
├── LICENSE
├── README.md
├── cluener                     #cluener数据集
│   ├── README.md
│   ├── cluener_predict.json
│   ├── dev.json
│   ├── test.json
│   └── train.json
├── main.py                        #训练代码
└── src
    ├── __init__.py
    ├── cluener_dataset.py        #cluener数据集处理代码
    ├── dataset.py                #数据集相应的处理代码
    ├── eval.py                   #模型评估代码
    ├── parameter.py              #各种超参数的设置，包括训练的各种参数，数据集，预训练模型等
    └── predict.py                #预测代码
```


# 二：项目效果

## (1) bert-base-multilingual-uncased 各数据集 测试集指标
dataset|loss|precision|recall|f1
--|-- |-- |-- |-- |
msra_ner|0.000102|0.911437|0.918944|0.915175
peoples_daily_ner|0.028725| 0.937808| 0.939352| 0.938580
cluener|0.602601| 0.730247|0.774975| 0.751947

## (2) bert-wwm-ext-chinese 各数据集 测试集指标
dataset|loss|precision|recall|f1
--|-- |-- |-- |-- |
msra_ner|0.004710|0.945492|0.938413|0.941939
peoples_daily_ner|0.020903|0.967318| 0.950329| 0.958749
cluener|0.433798|0.754693|0.790043| 0.771964

## (4) 运行的一些结果
预训练模型：bert-base-multilingual-uncased
数据集：peoples_daily_ner

```bash
在跨世纪的征途上，在中国共产党领导下，我们要努力实现包括各民主党派、各人民团体、无党派人士在内的全体中国人民的大团结，实现包括大陆同胞、台港澳同胞和海外侨胞在内的所有爱国的中华儿女的大团结，从而战胜各种艰难险阻，实现跨世纪的宏伟蓝图。
[['中国共产党', 10, 14, 'ORG'], ['中国', 50, 51, 'LOC'], ['大陆', 63, 64, 'LOC'], ['台', 68, 68, 'LOC'], ['台', 68, 68, 'LOC'], ['台', 68, 68, 'LOC'], ['中华', 86, 87, 'LOC']] 
======
中国共产党将坚定不移地贯彻“长期共存、互相监督、肝胆相照、荣辱与共”的方针，坚持和完善中国共产党领导的多党合作和政治协商制度，不断发展同各民主党派之间业已形成的真诚、有效的合作，不断推进政治协商、民主监督、参政议政的规范化和制度化，巩固中国共产党同各民
[['中国共产党', 0, 4, 'ORG'], ['中国共产党', 43, 47, 'ORG'], ['中国共产党', 118, 122, 'ORG']] 
======
主党派的联盟，充分发挥各民主党派在国家政治生活和社会生活中的作用，共同把建设有中国特色社会主义伟大事业推向新世纪。
[['联', 4, 4, 'ORG'], ['中国', 39, 40, 'LOC']] 
======
今年７月１日我国政府恢复对香港行使主权，标志着“一国两制”构想的巨大成功，标志着中国人民在祖国统一大业的道路上迈出了重要的一步。
[['香港', 13, 14, 'LOC'], ['中国', 40, 41, 'LOC']] 
======
实现祖国的完全统一，是海内外全体中华儿女的共同心愿，也是历史赋予我们的重任。
[['中华', 16, 17, 'LOC']] 
======
我们热诚希望致公党在今后的工作中，充分发挥与海外联系广泛的优势，多渠道、多层次、多形式地开展海外联络工作，积极宣传邓小平“一国两制”的科学构想和江泽民同志关于台湾问题的八项主张，为保持香港的繁荣稳定，为澳门的顺利回归和促进祖国完全统一作出新的更大的贡献
[['致公党', 6, 8, 'ORG'], ['邓小平', 57, 59, 'PER'], ['江泽民', 72, 74, 'PER'], ['台湾', 79, 80, 'LOC'], ['香港', 92, 93, 'LOC'], ['澳门', 101, 102, 'LOC']] 
======
致公党第十一次全国代表大会是致公党历史上一次重要的会议。
[['致公党第十一次全国代表大会', 0, 12, 'ORG'], ['致公党', 14, 16, 'ORG']] 
======
大会将完成致公党中央领导集体跨世纪的新老交替，确定今后一个时期的工作任务。
[['致公党中', 5, 8, 'ORG']] 
======
我们相信，面临新的形势任务，这次大会即将选出的新一届致公党中央领导集体，一定能够在实现新老交替的基础上完成政治交接，把老一辈领导人同中国共产党亲密合作的优良传统继承下来并发扬光大，切实加强自身建设，不断提高致公党的整体素质，充分发挥参政党作用，团结和带
[['致公党中央', 26, 30, 'ORG'], ['中国共产党', 66, 70, 'ORG'], ['致公党', 103, 105, 'ORG']] 
======
领广大成员和所联系的归侨、侨眷，埋头苦干，扎实工作，在建设有中国特色社会主义伟大事业中创造出新的业绩。
[['中国', 30, 31, 'LOC']] 
======
让我们在邓小平理论的指引下，更加紧密地团结在以江泽民同志为核心的中共中央周围，以高度的历史责任感和时代紧迫感，积极投身于改革开放和社会主义现代化建设的伟大实践，共同谱写我国社会主义现代化建设的新篇章，迎接辉煌的２１世纪！
[['邓小平', 4, 6, 'PER'], ['江泽民', 23, 25, 'PER'], ['中共中央', 32, 35, 'ORG']] 
======
祝中国致公党第十一次全国代表大会圆满成功！
[['中国致公党第十一次全国代表大会', 1, 15, 'ORG']] 
======
```

# 三：运行

训练的模型参数：
链接：https://pan.baidu.com/s/1hdfilBRyVQ3GkKOCOPSqng 
提取码：fpfi 

## （1）相关环境
1.安装 **Anacodna**,上面百度云盘分享链接，有win10 64位的annaconda安装包


2.安装相应第三方库
```bash
进入anaconda的python环境
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlepaddle
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paddlenlp
```

3.克隆项目

```bash
git clone https://gitee.com/lingcb/paddle-ner.git

cd paddle-ner

python src/predict.py      #预测
python src/eval.py         #评估模型
python main.py             #训练模型
```

## （2）运行
在运行上面项目之前，需要设置一些**parameter.py文件里的一些参数**

### 1.训练

```bash
cd paddle-ner
python main.py
```

**需要修改，下图的几个参数**，当然，也可结合实际，修改一些其他的参数。
![请添加图片描述](https://img-blog.csdnimg.cn/d5e89bb5b7784577b6486cd20111a4eb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54u45LiN5Yeh,size_20,color_FFFFFF,t_70,g_se,x_16)
![请添加图片描述](https://img-blog.csdnimg.cn/f988f2e9a19a43909a5432a3dcd749f1.png)
![请添加图片描述](https://img-blog.csdnimg.cn/71c16cfa6ad6465e9f7cdc9c18cf03d1.png)
运行成功，如下图
![请添加图片描述](https://img-blog.csdnimg.cn/32db3d0fd28545f1bb278531e52af1dc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54u45LiN5Yeh,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.预测&评估

```bash
cd paddle-ner
python src/predict.py      #预测
python src/eval.py         #评估模型
```

训练与评估需要改的参数，相似，其中**checkpoint_base**是上面**百度链接文件夹的路径**
![请添加图片描述](https://img-blog.csdnimg.cn/547c8e6874c84284b37cb4e92c6505a2.png)
![请添加图片描述](https://img-blog.csdnimg.cn/43e1979e6c684cbfbf72da3ff67bb4f0.png)
![请添加图片描述](https://img-blog.csdnimg.cn/90652b4ed3fb415b942c398766274ed0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54u45LiN5Yeh,size_20,color_FFFFFF,t_70,g_se,x_16)
预测代码成功运行截图
![请添加图片描述](https://img-blog.csdnimg.cn/6912957fed71484bb63e440a477fa51e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA54u45LiN5Yeh,size_20,color_FFFFFF,t_70,g_se,x_16)
指标代码成功运行截图



# 四：一些链接
（1）本工程代码开源链接：https://gitee.com/lingcb/paddle-ner
（2）paddlenlp 官方文档：https://paddlenlp.readthedocs.io/zh/latest/

（3）paddlenlp提供的预训练模型列表：https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html

（4）#CLUENER2020数据集开源链接
https://github.com/CLUEbenchmark/CLUENER2020

（5）paddlenlp datasets
https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html

# 后言
在做这个项目时，遇到了一些问题，这里记录一下。
1.paddlenlp.datasets.MapDataset,传入的是一个list,之前我传入一个paddle.io.Dataset，时，训练一段时间后，**会报 key:8 的错误**

2.cluener_dataset.py 代码，是处理原始cluener数据集的代码

3.dataset.py 里的**transform(texts)** 使用 里面的 **check_text_size(text,max_seq_len)** 进行最大长度序列切分，超过max_seq_len的部分，将作为**新的文本序列**

4.predict.py 预测前，，使用 dataset.py 里的**transform(texts)**，对原始文本texts数组进行预处理