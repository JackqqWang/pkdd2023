
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import os.path
from torch.utils.data import DataLoader, Dataset

def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_img(net_g, data_loader, args):

    test_loss = 0
    correct = 0
    
    for idx, (data, target) in enumerate(data_loader):
        # print("")
        if torch.cuda.is_available():
            data, target = data.to(args.device), target.to(args.device)
        log_probs= net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss
