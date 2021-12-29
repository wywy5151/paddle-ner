import os
import json
from paddle.io import Dataset


try:
    from src import parameter
except:
    import parameter
    
#生成 cluenerlabel_list  
def cluener_label_list():
    label_list = []
    for i in parameter.cluener_label:
        label_list.append("B-"+i)
        label_list.append("I-"+i)
    label_list.append("O")
    return label_list

label_list = cluener_label_list()
path = parameter.cluener_path


#加载文件数据
def load_data(path,name):
    with open(os.path.join(path,name),encoding="utf-8") as f:
        data = []
        line = f.readline()
        while line:
            data.append(json.loads(line))
            line = f.readline()
    return data


#转成BIO
def targeLabel(item):
    tokens = list(item["text"])
    labels = ["O"]*len(tokens)
    
    for name in item["label"]:
        for i in item["label"][name]:
            for j in item["label"][name][i]:
                #print(name,i,j)
                labels[j[0]] = "B-"+name
                for k in range(j[0]+1,j[-1]+1):
                    labels[k] = "I-"+name
    
    return {"tokens":tokens,"labels":labels}


#加载cluener数据集
def load_cluener(path):
    
    train = load_data(path,"train.json")
    test = load_data(path,"test.json")
    dev = load_data(path,"dev.json")
    
    train = [targeLabel(item) for item in train]
    dev = [targeLabel(item) for item in dev]
    
    return train,dev,test


# define a random dataset
class CluenerDataset(Dataset):
    def __init__(self,path,datatype="train"):
        
        self.train,self.dev,self.test = load_cluener(path)
        self.datatype = datatype
        
        
        self.label_list = label_list
        #print(label_list)
        self.id2label = dict(enumerate(self.label_list))
        self.label2id = dict(zip(self.label_list,range(len(self.label_list))))
        
        self.train_max_len = max([len(i["tokens"]) for i in self.train])
        self.dev_max_len = max([len(i["tokens"]) for i in self.dev])
        
        
    def __getitem__(self, idx):
        
        if self.datatype=="train":
            item = self.train[idx]
            item["labels"] = [self.label2id[i] for i in item["labels"]]
            item["tokens"] = item["tokens"]+["0"]*(self.train_max_len-len(item["tokens"]))
            item["labels"] = item["labels"]+[parameter.no_entity_id]*(self.train_max_len-len(item["labels"]))
        
        elif self.datatype=="dev":
            item = self.dev[idx]
            item["labels"] = [self.label2id[i] for i in item["labels"]]
            item["tokens"] = item["tokens"]+["0"]*(self.dev_max_len-len(item["tokens"]))
            item["labels"] = item["labels"]+[parameter.no_entity_id]*(self.dev_max_len-len(item["labels"]))
                   
        else:
            return self.test[idx]
        
        return item

    def __len__(self):
        if self.datatype=="train":
            return len(self.train)
        elif self.datatype=="dev":
            return len(self.dev)-1
        else:
            return len(self.test)-1


        
        
#a = CluenerDataset("D:/yunpan/数据集/cluener_public","dev")
        
        
        
        
        
        
        
        
        
        