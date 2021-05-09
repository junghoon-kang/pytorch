import torchmetrics
from torchmetrics import Metric


if __name__ == '__main__':
    import torch

    def Accuracy_test():
        accuracy = torchmetrics.Accuracy()
        trues = torch.tensor([0,1,2,3])
        preds = torch.tensor([0,2,1,3])
        accuracy.update(preds, trues)
        print(accuracy.compute())  # tensor(0.5000)
        trues = torch.tensor([0,2,1,3])
        preds = torch.tensor([0,2,1,3])
        accuracy.update(preds, trues)
        print(accuracy.compute())  # tensor(0.7500)
    #Accuracy_test()

    def Recall_test():
        preds = torch.tensor([2,0,2,1])
        trues = torch.tensor([1,1,2,0])
        recall = torchmetrics.Recall(average='macro', num_classes=3)
        print(recall(preds, trues))  # tensor(0.3333)
        recall = torchmetrics.Recall(average='micro')
        print(recall(preds, trues))  # tensor(0.2500)
        """
          0 1 2
        0 0 1 0
        1 1 0 1
        2 0 0 1
        """
    #Recall_test()
