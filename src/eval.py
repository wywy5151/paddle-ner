import os
import time
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification
try:
    from src import dataset
    from src import parameter
    import src.cluener_dataset as cluener
except:
    import dataset
    import parameter
    import cluener_dataset as cluener

from paddlenlp.datasets import MapDataset

max_steps = parameter.max_steps
num_train_epochs = parameter.num_train_epochs
learning_rate = parameter.learning_rate
warmup_steps = parameter.warmup_steps 


global_step = parameter.global_step
logging_steps = parameter.logging_steps
save_steps = parameter.save_steps
output_dir = parameter.output_dir

train_ds,test_ds = cluener.CluenerDataset(parameter.cluener_path,"train"),cluener.CluenerDataset(parameter.cluener_path,"dev")
train_ds = MapDataset(list(train_ds))
test_ds = MapDataset(list(test_ds))

label_list = cluener.label_list  
#dataloader
train_loader = dataset.create_dataloader(train_ds)
test_loader = dataset.create_dataloader(test_ds)


from tqdm import tqdm
for epoch in range(num_train_epochs):
    for step, batch in tqdm(enumerate(train_loader)):
        global_step += 1
        print("epoch=",epoch," global_step=",global_step)
        input_ids, token_type_ids, _, labels = batch
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    