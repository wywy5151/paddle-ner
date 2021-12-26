import os
import json

try:
    from src import parameter
    from src import dataset
except:
    import parameter
    
    
    
def load(fname):
    tokens = []
    texts = []
    labels = []
    
    with open(fname, "r", encoding='utf8') as f:
        text = ""
        token = []
        label = []
        
        for line in f.readlines():
            line = line.strip()
            if line != "":
                span_list = line.split('\t')
                raw_char = ''.join(list(span_list[0])[:-1])
                tag = span_list[-1]   
            else:
                pass
    
    

