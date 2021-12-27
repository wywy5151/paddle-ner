
import paddle
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer

try:
    from src import parameter
    from src import dataset
except:
    import parameter
    import dataset


label_list = parameter.msra_label_list
id2label = dict(enumerate(label_list))
label2id = dict(zip(label_list,range(len(label_list))))

def load_model(checkpoint_path):
    
    model = BertForTokenClassification.from_pretrained(parameter.pretrained, num_classes=parameter.num_classes)
    model_dict = paddle.load(checkpoint_path)
    model.set_dict(model_dict)
    model.eval()
    
    return model

def predict(texts,model):
    
    input_ids,token_type_ids,seq_len = dataset.transform(texts)
    input_ids = paddle.to_tensor(input_ids,dtype="int64")
    token_type_ids = paddle.to_tensor(token_type_ids,dtype="int64")
    logits = model(input_ids, token_type_ids)
    preds = logits.argmax(axis=2)
    
    return preds.numpy(),seq_len
    

def parse(outputs,seq_len,texts):
    results = []
    for i in range(len(outputs)):
        label = []
        result = []
        begin = 0
        end =   0
        word = ""
        for j in range(len(texts[i])):
            label.append(id2label[outputs[i][j]])
            if id2label[outputs[i][j]][0]=="B":
                if word:
                    result.append([word,begin,end,id2label[outputs[i][j]].split("-")[-1]])
                else:
                    begin = end = j
                    word=texts[i][j]
                    
            elif id2label[outputs[i][j]][0]=="I":
                word+=texts[i][j]
                end+=1
            else:
                if word:
                    result.append([word,begin,end,id2label[outputs[i][j]].split("-")[-1]])
                    word=""
        results.append([texts[i],label,result])
    return results
                    
                
        
    
model = load_model("D:/yunpan/checkpoint/model_22000.pdparams")

train_ds, test_ds = load_dataset("msra_ner", splits=["train", "test"])
texts = [train_ds.__getitem__(i)["tokens"] for i in range(10)]
preds,seq_len = predict(texts,model)


results = parse(preds,seq_len,texts)

print(results)












'''
init_checkpoint_path = 'C:/Users/Administrator/Desktop/paddle-ner/checkpoint/model_26080.pdparams'
model_name_or_path = 'bert-base-multilingual-uncased'


people_train,people_test,people_dev = load_dataset('peoples_daily_ner', splits=["train", "test","dev"])
train_ds, test_ds = people_train, people_dev
label_list = train_ds.label_list


tokenizer = BertTokenizer.from_pretrained(parameter.pretrained)
train_loader,test_loader = dataset.make_dataloader(train_ds,test_ds,label_list,tokenizer)


model = BertForTokenClassification.from_pretrained(model_name_or_path, num_classes=len(label_list))
if init_checkpoint_path:
    model_dict = paddle.load(init_checkpoint_path)
    model.set_dict(model_dict)
model.eval()


# Define the model netword and its loss
loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=parameter.ignore_label)
metric = ChunkEvaluator(label_list=label_list)


metric.reset()
for step, batch in enumerate(test_loader):
    input_ids, token_type_ids, length, labels = batch
    logits = model(input_ids, token_type_ids)
    loss = loss_fct(logits, labels)
    avg_loss = paddle.mean(loss)
    preds = logits.argmax(axis=2)
    num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
        length, preds, labels)
    metric.update(num_infer_chunks.numpy(),
                    num_label_chunks.numpy(), num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
        (avg_loss, precision, recall, f1_score))
    break
    

def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = "".join(input_words[idx]['tokens'])
        tags = [id2label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':
                if len(words):
                    sent_out.append(words)
                if t.startswith('B-'):
                    tags_out.append(t.split('-')[1])
                else:
                    tags_out.append(t)
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


raw_data = test_ds.data

id2label = dict(enumerate(test_ds.label_list))



pred_list = []
len_list = []
for step, batch in enumerate(test_loader):
    input_ids, token_type_ids, length, labels = batch
    logits = model(input_ids, token_type_ids)
    pred = paddle.argmax(logits, axis=-1)
    pred_list.append(pred.numpy())
    len_list.append(length.numpy())


preds = parse_decodes(raw_data, id2label, pred_list, len_list)
'''
