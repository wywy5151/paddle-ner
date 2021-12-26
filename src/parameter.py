
#预训练模型
pretrained = "bert-base-multilingual-uncased"
#模型输出路径
output_dir = 'data'


# Define the model netword and its loss
max_steps = -1
num_train_epochs = 10
learning_rate = 2e-5
warmup_steps = 1000
global_step = 0
logging_steps = 1
save_steps = 1000


#数据集
dataset = "msra_ner"  #msra_ner,peoples_daily_ner,cluener
ignore_label = -100
no_entity_id = 6  #“O” 在label_list的index,msra_ner:6 peoples_daily_ner:6 cluener:20  
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

