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
    model.train()