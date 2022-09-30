# import package

# model
import torch
import torch.nn as nn

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        return neg_Pearson_Loss(predictions, targets)

class neg_pear_loss(nn.Module):
    def __init__(self):
        super(neg_pear_loss, self).__init__()
        return

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):  # pred -> [batch x time_length]
            sum_x = torch.sum(preds[i])  # predict 배치의 i번째 요소의 합
            sum_y = torch.sum(labels[i])  # labels 배치의 i번째 요소의 합
            sum_xy = torch.sum(preds[i] * labels[i])  # *(아마다르 곱)
            sum_x_sq = torch.pow(preds[i], 2)
            sum_y_sq = torch.pow(labels[i], 2)
            t = preds.shape[1]

            num = t * sum_xy - sum_x * sum_y
            den = torch.sqrt((t * sum_x_sq - torch.pow(sum_x, 2)) * ((t * sum_y_sq) - torch.pow(sum_y, 2)))
            loss += 1 - num / den

        loss = loss / preds.shape[0]  # batch 평균 loss 반환
        return loss

def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    rst = 0
    targets = targets[:, :]
    predictions = torch.squeeze(predictions)

    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
    targets = (targets - torch.mean(targets)) / torch.std(targets)

    for i in range(predictions.shape[0]):
        sum_x = torch.sum(predictions[i])  # x
        sum_y = torch.sum(targets[i])  # y
        sum_xy = torch.sum(predictions[i] * targets[i])  # xy
        sum_x2 = torch.sum(torch.pow(predictions[i], 2))  # x^2
        sum_y2 = torch.sum(torch.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst