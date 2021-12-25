
#cluener数据集路径
base="D:/yunpan/数据集/cluener_public"

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
save_steps = 500
ignore_label = -100


#设置最长序列
max_seq_len=128
#训练集 batch_size
train_batch_size=8
#测试集 batch_size
test_batch_size=32
