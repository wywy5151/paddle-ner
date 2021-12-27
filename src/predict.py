
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification

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
    
    input_ids,token_type_ids,seq_len,texts = dataset.transform(texts)
    input_ids = paddle.to_tensor(input_ids,dtype="int64")
    token_type_ids = paddle.to_tensor(token_type_ids,dtype="int64")
    logits = model(input_ids, token_type_ids)
    preds = logits.argmax(axis=2)
    
    return preds.numpy(),seq_len,texts
    

def parse(outputs,seq_len,texts):
    results = []
    for i in range(len(outputs)):
        label = []
        result = []
        begin = 0
        end =   0
        word = ""
        texts[i] = "0"+texts[i]+"0"
        for j in range(len(texts[i])):
            label.append(id2label[outputs[i][j]])
            if id2label[outputs[i][j]][0]=="B":
                if word:
                    result.append([word,begin-1,end-1,id2label[outputs[i][j-1]].split("-")[-1]])
                else:
                    begin = end = j
                    word=texts[i][j]
                    
            elif id2label[outputs[i][j]][0]=="I":
                word+=texts[i][j]
                end+=1
            else:
                if word:
                    result.append([word,begin-1,end-1,id2label[outputs[i][j-1]].split("-")[-1]])
                    word=""
        results.append([texts[i][1:-1],label,result])
    return results
                    
                
if __name__ == "__main__":
    model = load_model("D:/yunpan/checkpoint/model_22000.pdparams")
    
    train_ds, test_ds = load_dataset("peoples_daily_ner", splits=["train", "test"])
    texts = ["".join(test_ds.__getitem__(i)["tokens"]) for i in range(10)]
    
    
    preds,seq_len,texts = predict(texts,model)
    results = parse(preds,seq_len,texts)
    
    for r in results:
        print(r[0])
        print(r[2],"\n======")
        


