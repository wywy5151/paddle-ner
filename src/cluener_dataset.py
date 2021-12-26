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
def load_data(name):
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
def load_cluener(base):
    
    train = load_data("train.json")
    test = load_data("test.json")
    dev = load_data("dev.json")
    
    train = [targeLabel(item) for item in train]
    dev = [targeLabel(item) for item in dev]
    
    return train,dev,test


# define a random dataset
class CluenerDataset(Dataset):
    def __init__(self,path,datatype="train"):
        
        self.train,self.dev,self.test = load_cluener(path)
        self.datatype = datatype
        
        self.label_list = label_list
        print(label_list)
        self.id2label = dict(enumerate(self.label_list))
        self.label2id = dict(zip(self.label_list,range(len(self.label_list))))

    def __getitem__(self, idx):
        
        if self.datatype=="train":
            item = self.train[idx]
        elif self.datatype=="dev":
            item = self.dev[idx]
        else:
            return self.test
        
        item["labels"] = [self.label2id[i] for i in item["labels"]]
        return item

    def __len__(self):
        if self.datatype=="train":
            return len(self.train)
        elif self.datatype=="dev":
            return len(self.dev)
        else:
            return len(self.test)






            
        
        
        
        
        
        
        
        
        
        
        
        
        
        