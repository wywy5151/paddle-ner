
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification

try:
    from src import parameter
    from src import dataset
    import src.cluener_dataset as cluener
except:
    import parameter
    import dataset
    import cluener_dataset as cluener

if parameter.dataset == "msra_ner":
    print(1)
    test_ds = load_dataset("msra_ner", splits=["test"])
    label_list = parameter.msra_label_list
elif parameter.dataset == "peoples_daily_ner":
    print(2)
    test_ds = load_dataset('peoples_daily_ner', splits=["dev"])
    label_list = parameter.peoples_daily_label_list
elif parameter.dataset == "cluener":
    print(3)
    test_ds = cluener.CluenerDataset(parameter.cluener_path,"dev")
    label_list = cluener.label_list  
else:
    print("请输入正确的数据集名！")
    exit(0)
    
    
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
        texts[i] = "00"+texts[i]+"0"
        texts[i] = texts[i]
        for j in range(len(texts[i])):
            label.append(id2label[outputs[i][j]])
            if id2label[outputs[i][j]][0]=="B":
                if word:
                    result.append([word,begin-1,end-1,id2label[outputs[i][j-1]].split("-")[-1]])
                    word=texts[i][j]
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
    
    checkpoint = {}
    checkpoint["msra_ner"] = parameter.msra_ner_checkpoint
    checkpoint["peoples_daily_ner"] = parameter.peoples_daily_ner_checkpoint
    checkpoint["cluener"] = parameter.cluener_checkpoint
    
    model = load_model(checkpoint[parameter.dataset])
    
    #原始文本
    texts = ["".join(test_ds[i+10]["tokens"]) for i in range(10)]
    
    preds,seq_len,texts = predict(texts,model)
    results = parse(preds,seq_len,texts)
    
    for r in results:
        print(r[0])
        print(r[2],"\n======")
        


