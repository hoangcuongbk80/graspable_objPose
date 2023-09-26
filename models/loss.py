import torch.nn as nn
import torch


def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspability_loss, end_points = compute_graspability_loss(end_points)
    view_loss, end_points = compute_view_graspability_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    width_loss, end_points = compute_width_loss(end_points)
    loss = objectness_loss + 10 * graspability_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspability_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspability_score = end_points['graspability_score'].squeeze(1)
    graspability_label = end_points['graspability_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspability_score, graspability_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    
    graspability_score_c = graspability_score.detach().clone()[loss_mask]
    graspability_label_c = graspability_label.detach().clone()[loss_mask]
    graspability_score_c = torch.clamp(graspability_score_c, 0., 0.99)
    graspability_label_c = torch.clamp(graspability_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspability_score_c * 20) - torch.trunc(graspability_label_c * 20)) / 20.).mean()
    end_points['stage1_graspability_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspability_loss'] = loss
    return loss, end_points


def compute_view_graspability_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspability']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
