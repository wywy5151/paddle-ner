import os
import time
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification

from src import dataset
from src import parameter

import src.cluener_dataset as cluener
from paddlenlp.datasets import MapDataset


#加载数据集
if parameter.dataset == "msra_ner":
    print(1)
    train_ds, test_ds = load_dataset("msra_ner", splits=["train", "test"])
    label_list = parameter.msra_label_list
elif parameter.dataset == "peoples_daily_ner":
    print(2)
    train_ds, test_ds = load_dataset('peoples_daily_ner', splits=["train","dev"])
    label_list = parameter.peoples_daily_label_list
elif parameter.dataset == "cluener":
    print(3)
    train_ds,test_ds = cluener.CluenerDataset(parameter.cluener_path,"train"),cluener.CluenerDataset(parameter.cluener_path,"dev")
    train_ds = MapDataset(list(train_ds))
    test_ds = MapDataset(list(test_ds))
    label_list = cluener.label_list  
else:
    print("请输入正确的数据集名！")
    exit(0)
    

#dataloader
train_loader = dataset.create_dataloader(train_ds)
test_loader = dataset.create_dataloader(test_ds)


# Define the model netword and its loss
max_steps = parameter.max_steps
num_train_epochs = parameter.num_train_epochs
learning_rate = parameter.learning_rate
warmup_steps = parameter.warmup_steps 


global_step = parameter.global_step
logging_steps = parameter.logging_steps
save_steps = parameter.save_steps
output_dir = parameter.output_dir

#加载预训练模型
model = BertForTokenClassification.from_pretrained(parameter.pretrained, num_classes=len(label_list))

num_training_steps = max_steps if max_steps > 0 else len(train_loader) * num_train_epochs

lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_steps)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]


optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    epsilon=1e-8,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)


loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=parameter.ignore_label)
metric = ChunkEvaluator(label_list=label_list)


def evaluate(model, loss_fct, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            None, length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()
    

#paddle.set_device('gpu')
#if paddle.distributed.get_world_size() > 1:
#    paddle.distributed.init_parallel_env()
    
    
last_step = num_train_epochs * len(train_loader)
tic_train = time.time()

for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_loader):
        global_step += 1
        input_ids, token_type_ids, _, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        
        
        if global_step % logging_steps == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (global_step, epoch, step, avg_loss,
                    logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
            
            
        #反向传播并更新参数
        avg_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
        
        
        #保存模型
        if global_step % save_steps == 0 or global_step == last_step:
            if paddle.distributed.get_rank() == 0:
                evaluate(model, loss_fct, metric, test_loader,len(label_list))
                paddle.save(model.state_dict(),
                            os.path.join(output_dir,
                                            "model_%d.pdparams" % global_step))
                
paddle.save(model.state_dict(),
            os.path.join(output_dir,
                            "model_%d.pdparams" % global_step))
















