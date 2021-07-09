import torch

# loss function


def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    try:
        label = label.long()
        mask = (label != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        #print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduction='none')
        return torch.sum(cost)
    except ValueError:

        print('label size: ', label.size())
        print('predict size: ', prediction.size())

