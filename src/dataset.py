import paddle
from functools import partial
from paddle.io import DataLoader
from paddlenlp.data import Stack, Pad, Dict
from paddlenlp.transformers import BertTokenizer


try:
    from src import parameter
except:
    import parameter


tokenizer = BertTokenizer.from_pretrained(parameter.pretrained)   
ignore_label = parameter.ignore_label
no_entity_id = parameter.no_entity_id


# 将文本转成 paddlenlp bert token
def tokenize_and_align_labels(example, tokenizer, no_entity_id,max_seq_len=512):
    
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP] 剔除
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    
    return tokenized_input


#初始化一些默认参数
trans_func = partial(
    tokenize_and_align_labels,
    tokenizer=tokenizer,
    no_entity_id=no_entity_id,
    max_seq_len=parameter.max_seq_len) 


#对序列进行预处理
batchify_fn = lambda samples, fn=Dict({
    'input_ids': Pad(dtype="int64",axis=0, pad_val=tokenizer.pad_token_id),            # input
    'token_type_ids': Pad(dtype="int64",axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    'seq_len': Stack(dtype="int64"),  # seq_len
    'labels': Pad(dtype="int64",axis=0, pad_val=ignore_label)                          # label
}): fn(samples)


def create_dataloader(ds):
    #文本转token
    ds = ds.map(trans_func)
    #配置数据集加载参数
    batch_sampler = paddle.io.DistributedBatchSampler(ds, batch_size=parameter.train_batch_size, shuffle=True, drop_last=True)
    
    #训练集加载器
    data_loader = DataLoader(
        dataset=ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_sampler=batch_sampler,
        return_list=True)

    return data_loader


#############################################################################

def check_text_size(text,max_seq_len):
    
    split = []
    if len(text)<=max_seq_len:
        split = [text]
    else:
        count = len(text)//max_seq_len
        for i in range(count+1):
            split.append(text[i*max_seq_len:i*max_seq_len+max_seq_len])
    
    if not split[-1]:
        split = split[:-1]
        
    return split

def transform(texts):
    """
    如果文本长度大于parameter.max_seq_len
    则，切分文本到

    """
    input_ids = []
    token_type_ids = []
    seq_len = []
    
    split_texts = []
    for t in texts:
        split_texts += check_text_size(t,parameter.max_seq_len-2)

    for text in split_texts:
        tokenized_input = tokenizer(
            text,
            return_length=True,
            is_split_into_words=True,
            max_seq_len=parameter.max_seq_len)
        
        input_ids.append(tokenized_input["input_ids"])
        token_type_ids.append(tokenized_input["token_type_ids"])
        seq_len.append(tokenized_input["seq_len"])
    
    maxlen = max(seq_len)
   
    inputs = []
    types = []
    for i in range(len(split_texts)):
        inputs.append([0]+input_ids[i]+[0]*(maxlen-seq_len[i]+3))
        types.append([0]+token_type_ids[i]+[0]*(maxlen-seq_len[i]+3))
        
    
    return inputs,types,seq_len,split_texts


        


        
        
        