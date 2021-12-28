
import paddle
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.datasets import MapDataset


try:
    from src import predict
    from src import dataset
    from src import parameter
    import src.cluener_dataset as cluener
except:
    import predict
    import dataset
    import parameter
    import cluener_dataset as cluener


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
    
    test_ds = list(test_ds)
    max_seq_len = max([len(i["tokens"]) for i in test_ds])
    
    test_ds = MapDataset(list(test_ds))
    label_list = cluener.label_list  
else:
    print("请输入正确的数据集名！")
    exit(0)
    

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
    
    return avg_loss, precision, recall, f1_score
    
    
if __name__ == "__main__":
    
    checkpoint = {}
    checkpoint["msra_ner"] = parameter.msra_ner_checkpoint
    checkpoint["peoples_daily_ner"] = parameter.peoples_daily_ner_checkpoint
    checkpoint["cluener"] = parameter.cluener_checkpoint
    
    test_loader = dataset.create_dataloader(test_ds)
    model = predict.load_model(checkpoint[parameter.dataset])
    
    avg_loss, precision, recall, f1_score = evaluate(model, loss_fct, metric, test_loader, len(label_list))
    
    
    
    
