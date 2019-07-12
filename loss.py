import torch
import torch.nn.functional as F
from fastai.callback import Callback, add_metrics

def loss_fn(output, target, meta_target):
    META_BETA = 10.0
    pred, meta_pred = output[0], output[1]
    target_loss = F.cross_entropy(pred, target)
    meta_pred = torch.chunk(meta_pred, chunks=12, dim=1)
    meta_target = torch.chunk(meta_target, chunks=12, dim=1)
    meta_target_loss = 0.
    for idx in range(0, 12):
        meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_pred[idx]), meta_target[idx])
    loss = target_loss + META_BETA*meta_target_loss / 12.
    return loss

class Accuracy(Callback):
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output[0].argmax(1)
        last_target = last_target[0]
        self.correct += preds.eq(last_target).sum().float()
        self.total += last_target.size()[0]

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, 100 * self.correct/self.total)

