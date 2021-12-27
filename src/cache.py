
import paddle
from functools import partial
from paddle.io import DataLoader
from paddlenlp.data import Stack, Pad, Dict

try:
    from src import parameter
except:
    import parameter


base = parameter.cluener_path



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



def make_dataloader(train_ds,test_ds,label_list,tokenizer):
    
    label_num = len(label_list)       #标签数
    no_entity_id = label_num - 1       

    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=parameter.max_seq_len)

    train_ds = train_ds.map(trans_func)
    ignore_label = parameter.ignore_label

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(dtype="int64",axis=0, pad_val=tokenizer.pad_token_id),            # input
        'token_type_ids': Pad(dtype="int64",axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        'seq_len': Stack(dtype="int64"),  # seq_len
        'labels': Pad(dtype="int64",axis=0, pad_val=ignore_label)                          # label
    }): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=parameter.train_batch_size, shuffle=True, drop_last=True)
    
    #训练集加载器
    train_data_loader = DataLoader(
        dataset=train_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    #测试集加载器
    test_ds = test_ds.map(trans_func)
    test_data_loader = DataLoader(
            dataset=test_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_size=parameter.test_batch_size,
            return_list=True)
    
    return train_data_loader,test_data_loader



from paddlenlp.datasets import load_dataset
train_ds, test_ds = load_dataset("msra_ner", splits=["train", "test"])