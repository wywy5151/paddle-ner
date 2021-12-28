
#预训练模型
pretrained = "bert-wwm-ext-chinese"            # bert-base-multilingual-uncased  其他预训练模型可参考：https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html#transformer
pretrained = "bert-base-multilingual-uncased"


#模型输出路径
output_dir = 'cluener'


# Define the model netword and its loss
max_steps = -1
num_train_epochs = 10
learning_rate = 2e-5
warmup_steps = 1000
global_step = 0
logging_steps = 1
save_steps = 10000


#数据集
dataset = "cluener"  #msra_ner,peoples_daily_ner,cluener
ignore_label = -100
no_entity_id = 20  #“O” 在label_list的index,msra_ner:6 peoples_daily_ner:6 cluener:20  
num_classes = 21     #num_classes=len(label_list),msra_ner:7 peoples_daily_ner:7 cluener:21  
#设置最长序列
max_seq_len=128
#训练集 batch_size
train_batch_size=8
#测试集 batch_size
test_batch_size=32


#cluener 标签
cluener_label = {"address":"地址","book":"书名","company":"公司","game":"游戏","government":"政府","movie":"电影","name":"姓名","organization":"组织机构","position":"职位","scene":"景点"}
#cluener数据集路径
cluener_path = "D:/yunpan/数据集/cluener_public"


#label list
cluener_label_list = ['B-address', 'I-address', 'B-book', 'I-book', 'B-company', 'I-company', 'B-game', 'I-game', 'B-government', 'I-government', 'B-movie', 'I-movie', 'B-name', 'I-name', 'B-organization', 'I-organization', 'B-position', 'I-position', 'B-scene', 'I-scene', 'O']
msra_label_list = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
peoples_daily_label_list = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']


