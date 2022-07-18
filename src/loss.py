import torch.nn.functional as F
import torch


def l1_loss(pred, gt, is_valid=None):
    loss = F.l1_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()

def mse_loss(pred, gt, is_valid=None):
    loss = F.mse_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()


def bce_loss(pred, gt):
    return F.binary_cross_entropy(pred, gt, reduction='mean')


def normal_loss(pred, gt, face, is_valid=None):

    v1_out = pred[:, face[:, 1], :] - pred[:, face[:, 0], :]
    v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
    v2_out = pred[:, face[:, 2], :] - pred[:, face[:, 0], :]
    v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
    v3_out = pred[:, face[:, 2], :] - pred[:, face[:, 1], :]
    v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

    v1_gt = gt[:, face[:, 1], :] - gt[:, face[:, 0], :]
    v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
    v2_gt = gt[:, face[:, 2], :] - gt[:, face[:, 0], :]
    v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
    normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
    normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

    # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) #* valid_mask
    cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) #* valid_mask
    cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) #* valid_mask
    loss = torch.cat((cos1, cos2, cos3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()


def edge_length_loss(pred, gt, face, is_valid=None):

    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
    # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
    # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    diff1 = torch.abs(d1_out - d1_gt) #* valid_mask_1
    diff2 = torch.abs(d2_out - d2_gt) #* valid_mask_2
    diff3 = torch.abs(d3_out - d3_gt) #* valid_mask_3
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        loss *= is_valid
    return loss.mean()

def prepare_ranking_data_user_agnostic(confidence, l1_distances):
    """
    Args:
        confidence: (batch_size,)
        l1_distance: (batch_sixe, )
    
    Returns:
        x1:         (batch_size choose 2, ), confidence values
        x2:         (batch_size choose 2, ), confidence values
        y:      (batch_size choose 2, ), the ranking target

    see https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html for more
    detailed explanation on xs and y.
    """
    bs = confidence.shape[0]
    indices = torch.arange(bs)
    idx_pairs = torch.combinations(indices)
    # data_pairs = confidence[idx_pairs, 0]
    x1 = confidence[idx_pairs[:,0]]
    x2 = confidence[idx_pairs[:,1]]

    l1 = l1_distances[idx_pairs[:,0]]
    l2 = l1_distances[idx_pairs[:,1]]
    y = torch.sign(l2 - l1)
    return x1, x2, y

def prepare_ranking_data(confidence, l1_distances, user_ids=None, user_wise=False):
    """
    Args:
        confidence: (batch_size,)
        l1_distance: (batch_sixe, )
    
    Returns:
        x1:         (batch_size choose 2, ), confidence values
        x2:         (batch_size choose 2, ), confidence values
        y:      (batch_size choose 2, ), the ranking target

    see https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html for more
    detailed explanation on xs and y.
    """
    # assert len(user_ids.shape) == 1
    # print(user_ids.shape)
    if not user_wise:
        return prepare_ranking_data_user_agnostic(confidence, l1_distances)
    
    else:
        x1 = []
        x2 = []
        y = []
        unique_ids, counts = torch.unique(user_ids, return_counts=True)
        print('HERE'*10)
        for i, user_id in enumerate(unique_ids):
            if counts[i] > 1:
                mask = user_ids == user_id
                x1_temp, x2_temp, y_temp = prepare_ranking_data_user_agnostic(confidence[mask], l1_distances[mask])
                x1.append(x1_temp)
                x2.append(x2_temp)
                y.append(y_temp)

        if (len(x1)) < 1:
            return confidence[:-1], confidence[1:], torch.sign(l1_distances[1:] - l1_distances[:-1])
            
        return torch.cat(x1), torch.cat(x2), torch.cat(y)