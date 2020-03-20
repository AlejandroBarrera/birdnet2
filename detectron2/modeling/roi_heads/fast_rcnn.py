# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import batched_nms, batched_nms_rotated, nms_rotated, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import math
logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, vp_bins=None, viewpoint=None, viewpoint_residual=None, rotated_box_training=False, height=None):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    viewpoint = [None]*boxes[0].shape[0] if viewpoint is None else viewpoint
    viewpoint_residual = [None]*boxes[0].shape[0] if viewpoint_residual is None else viewpoint_residual
    height = [None]*boxes[0].shape[0] if height is None else height
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vp_bins, vp, vp_res, rotated_box_training, h
        )
        for scores_per_image, boxes_per_image, image_shape, vp, vp_res, h in zip(scores, boxes, image_shapes, viewpoint, viewpoint_residual, height)
    ]
    return tuple(list(x) for x in zip(*result_per_image))

def anglecorrection(angle): # We need to transform from KITTI angle to normal one to perform nms rotated
    result = -angle-90 if angle <= 90 else 270-angle
    return torch.tensor([result])

def bin2ang(bin, bins): 
    bin_dist = np.linspace(-180,180,bins+1) # 180 is considered as -180
    return anglecorrection(bin_dist[bin])

def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, vp_bins=None, vp=None, vp_res=None, rotated_box_training=False, h=None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    if not rotated_box_training or len(boxes)==0:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        # BBox with encoding ctr_x,ctr_y,w,l
        if vp is not None and vp_bins is not None:
            _vp = vp.view(-1, num_bbox_reg_classes,vp_bins) # R x C x bins
            _vp = _vp[filter_mask]
            if len(_vp)>0:
                _, vp_max = torch.max(_vp, 1)
                vp_filtered = vp_max
                if vp_res is not None:
                    _vp_res = vp_res.view(-1, num_bbox_reg_classes, vp_bins)
                    _vp_res = _vp_res[filter_mask]
                    vp_res_filtered=list()
                    for i,k in enumerate(vp_max):
                        vp_res_filtered.append(_vp_res[i,k])
                else:
                    vp_filtered = _vp
            rboxes = []
            for i in range(boxes.shape[0]):
                box = boxes[i]
                angle = anglecorrection(vp_res_filtered[i]*180/math.pi).to(box.device) if vp_res is not None else bin2ang(vp_filtered[i],vp_bins).to(box.device)
                box = torch.cat((box,angle))
                rboxes.append(box)
            rboxes = torch.cat(rboxes).reshape(-1,5).to(vp_filtered.device)
            #keep = nms_rotated(rboxes, scores, nms_thresh)
            keep = batched_nms_rotated(rboxes, scores, filter_inds[:, 1], nms_thresh)
        else:
            boxes[:,:,2] = boxes[:,:,2] + boxes[:,:,0] #x2
            boxes[:,:,3] = boxes[:,:,3] + boxes[:,:,1] #y2
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    if vp is not None and vp_bins is not None:
        vp = vp.view(-1, num_bbox_reg_classes,vp_bins) # R x C x bins
        vp = vp[filter_mask]
        vp = vp[keep]
        if vp_res is not None:
            vp_res = vp_res.view(-1, num_bbox_reg_classes, vp_bins)
            vp_res = vp_res[filter_mask]
            vp_res = vp_res[keep]
        if len(vp)>0:
            _, vp_max = torch.max(vp, 1)
            result.viewpoint = vp_max
            if vp_res is not None:
                vp_res_filtered=list()
                for i,k in enumerate(vp_max):
                    vp_res_filtered.append(vp_res[i,k])
                # This result is directly the yaw orientation predicted
                result.viewpoint_residual = torch.tensor(vp_res_filtered).to(vp_max.device)
        else:
            result.viewpoint = vp
            result.viewpoint_residual = vp_res
    if h is not None:
        h = h.view(-1, num_bbox_reg_classes,2) # R x C x bins
        h = h[filter_mask]
        h = h[keep]
        result.height = h
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, vp_bins=None, viewpoint_logits=None, viewpoint_res_logits=None, rotated_box_training=False, height_logits=None, weights_height=None
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.viewpoint_logits = viewpoint_logits
        self.viewpoint = True if viewpoint_logits is not None else False
        self.vp_bins = vp_bins
        self.viewpoint_res = True if viewpoint_res_logits is not None else False
        self.viewpoint_res_logits = viewpoint_res_logits
        self.rotated_box_training = rotated_box_training
        self.height_logits = height_logits
        self.height_training = True if height_logits is not None else False
        self.weights_height = weights_height

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            if self.rotated_box_training:
                self.gt_boxes = cat([p.gt_bbox3D for p in proposals])
            else:
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            if proposals[0].has("gt_viewpoint") and self.viewpoint:
                self.gt_viewpoint = cat([p.gt_viewpoint for p in proposals], dim=0)
                if proposals[0].has("gt_viewpoint_rads") and self.viewpoint_res:
                    self.gt_viewpoint_rads = cat([p.gt_viewpoint_rads for p in proposals], dim=0)
            if proposals[0].has("gt_height") and self.height_training:
                self.gt_height = cat([p.gt_height for p in proposals], dim=0)

    def _log_accuracy(self, gt, pred_logits):
        """
        Log the accuracy metrics to EventStorage.
        """
        report = ''
        if gt is self.gt_classes:
            report = 'cls'
        elif gt is self.gt_viewpoint:
            report = 'viewpoint'
        num_instances = gt.numel()
        pred_classes = pred_logits.argmax(dim=1)
        bg_class_ind = pred_logits.shape[1] - 1

        fg_inds = (gt >= 0) & (gt < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = gt[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == gt).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/{}_accuracy".format(report), num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_{}_accuracy".format(report), fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/{}_false_negative".format(report), num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        
        self._log_accuracy(self.gt_classes, self.pred_class_logits)
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def softmax_cross_entropy_loss_vp(self):

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(1)
        fg_gt_classes = self.gt_classes[fg_inds]
        vp_list = list()
        for idx,logit in enumerate(self.viewpoint_logits[fg_inds]):
            vp_list.append(logit[int(fg_gt_classes[idx])*self.vp_bins:int(fg_gt_classes[idx])*self.vp_bins+self.vp_bins]) #Theoricatly the last class is background... SEE bg_class_ind
                
        filtered_viewpoint_logits = torch.cat(vp_list).view(self.gt_viewpoint[fg_inds].size()[0],self.vp_bins)  

        self._log_accuracy(self.gt_viewpoint[fg_inds], filtered_viewpoint_logits)

        loss = F.cross_entropy(filtered_viewpoint_logits, self.gt_viewpoint[fg_inds], reduction="sum")
        loss = loss / self.gt_classes.numel()
        return loss 

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        self.gt_boxes = self.gt_boxes if self.rotated_box_training else self.gt_boxes.tensor
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes, self.rotated_box_training
        )
        # dx, dy, dw, dh
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def smooth_l1_loss_height(self):
        """
        Compute the smooth L1 loss for height regression.

        Returns:
            scalar Tensor
        """
        gt_height_deltas = self.get_h_deltas()
        # dh,dz
        box_dim = gt_height_deltas.size(1) 
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        fg_gt_classes = self.gt_classes[fg_inds]
        # 2 columns 
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
        loss_box_reg = smooth_l1_loss(
            self.height_logits[fg_inds[:, None], gt_class_cols],
            gt_height_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized as in box delta regression task
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def get_h_deltas(self):

        gt_height = self.gt_height
        gt_classes = self.gt_classes
        src_heights = torch.tensor([130.05, 149.6, 147.9, 1.0]).to(gt_classes.device) #Mean heights encoded 

        target_heights = gt_height[:, 0]
        # For ground codification
        # target_ground = gt_height[:, 1]
        # target_ctr = target_ground + 0.5*target_heights # target_ground NOT CODIFICATED
        target_ctr = gt_height[:, 1]

        wh, wg, wz = self.weights_height
        dh = wh * torch.log(target_heights / src_heights[gt_classes])
        # dg = wg * target_ground
        dz = wz * (target_ctr - src_heights[gt_classes]/2.) / src_heights[gt_classes]

        deltas = torch.stack((dh, dz), dim=1)
        return deltas

    def smooth_l1_loss_vp_residual(self):
        """
        Compute the smooth L1 loss for viewpoint regression.

        Returns:
            scalar Tensor
        """
        
        gt_vp_deltas = self.get_vp_deltas()
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        fg_gt_classes = self.gt_classes[fg_inds]
        # pdb.set_trace()
        res_index_list = list()
        for idx,logit in enumerate(self.viewpoint_res_logits[fg_inds]):
            res_index_list.append(fg_gt_classes[idx]*self.vp_bins+self.gt_viewpoint[fg_inds][idx])

        loss_box_reg = smooth_l1_loss(
            self.viewpoint_res_logits[fg_inds,res_index_list],
            gt_vp_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def get_vp_deltas(self):
        gt_viewpoint = self.gt_viewpoint
        bin_dist = np.linspace(-math.pi,math.pi,self.vp_bins+1)
        bin_res = (bin_dist[1]-bin_dist[0])/2.
        
        src_vp_res = torch.tensor(bin_dist-bin_res,dtype=torch.float32).to(self.pred_proposal_deltas.device)
        target_vp = self.gt_viewpoint_rads
        src_vp_proposals = src_vp_res[gt_viewpoint]
        src_vp_proposals[target_vp>src_vp_res[self.vp_bins]] = src_vp_res[self.vp_bins]

        wvp = np.trunc(1/bin_res)
        dvp = wvp * (target_vp - src_vp_proposals - bin_res)
        deltas = dvp
        return deltas

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss_dict = {
                "loss_cls": self.softmax_cross_entropy_loss(),
                "loss_box_reg": self.smooth_l1_loss(),
            }
        if self.viewpoint:
            loss_dict["loss_viewpoint"] = self.softmax_cross_entropy_loss_vp()
        if self.viewpoint_res:
            loss_dict["loss_viewpoint_residuals"] = self.smooth_l1_loss_vp_residual()
        if self.height_training:
            loss_dict["loss_height"] = self.smooth_l1_loss_height()
        return loss_dict

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
            self.rotated_box_training
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def predict_viewpoint(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted orientation probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.viewpoint_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def apply_vp_deltas(self):
        deltas = self.viewpoint_res_logits
        assert torch.isfinite(deltas).all().item()

        bin_dist = np.linspace(-math.pi,math.pi,self.vp_bins+1)
        bin_res = (bin_dist[1]-bin_dist[0])/2.
        bin_dist = bin_dist-bin_res
        bin_dist = np.tile(bin_dist[:-1],self.pred_class_logits.shape[1] - 1)
        src_vp_res = torch.tensor(bin_dist,dtype=torch.float32).to(self.pred_proposal_deltas.device)

        wvp = np.trunc(1/bin_res)

        dvp = deltas / wvp
        pred_vp_res = dvp + src_vp_res + bin_res
        pred_vp_res[pred_vp_res<-math.pi] += 2*math.pi

        return pred_vp_res

    def predict_viewpoint_residual(self):
        num_pred = len(self.proposals)
        vp_residuals = self.apply_vp_deltas()
        return vp_residuals.split(self.num_preds_per_image, dim=0)

    def apply_h_deltas(self):

        deltas = self.height_logits
        assert torch.isfinite(deltas).all().item()
        src_heights = torch.tensor([130.05, 149.6, 147.9]).to(deltas.device) #Without background class?

        wh, wg, wz = self.weights_height
        dh = deltas[:, 0::2] / wh
        # dg = deltas[:, 1::2] / wg
        dz = deltas[:, 1::2] / wz

        pred_h = torch.exp(dh) * src_heights #Every class multiplied by every mean height
        # pred_g = dg
        pred_z = dz * src_heights + src_heights/2.

        pred_height = torch.zeros_like(deltas)
        pred_height[:, 0::2] = pred_h
        # pred_height[:, 1::2] = pred_g 
        pred_height[:, 1::2] = pred_z 
        return pred_height

    def predict_height(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic heights and elevations
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is 2
        """
        # pdb.set_trace()
        num_pred = len(self.proposals)
        B = 2
        K = self.height_logits.shape[1] // B
        heights = self.apply_h_deltas()
        return heights.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        viewpoint = self.predict_viewpoint() if self.viewpoint else None
        viewpoint_residual = self.predict_viewpoint_residual() if self.viewpoint_res else None
        height = self.predict_height() if self.height_training else None

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, self.vp_bins, viewpoint, viewpoint_residual, self.rotated_box_training, height
        )


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.viewpoint = cfg.VIEWPOINT
        viewpoint_bins = cfg.VP_BINS
        self.viewpoint_residual = cfg.VIEWPOINT_RESIDUAL
        self.height_training = cfg.HEIGHT_TRAINING
        if self.viewpoint:
            self.viewpoint_pred = nn.Linear(input_size, viewpoint_bins * num_classes)
            # torch.nn.init.kaiming_normal_(self.viewpoint_pred.weight,nonlinearity='relu')
            nn.init.xavier_normal_(self.viewpoint_pred.weight)
            nn.init.constant_(self.viewpoint_pred.bias, 0)
        if self.viewpoint_residual:
            self.viewpoint_pred_residuals = nn.Linear(input_size, viewpoint_bins * num_classes)
            nn.init.xavier_normal_(self.viewpoint_pred_residuals.weight)
            # torch.nn.init.kaiming_normal_(self.viewpoint_pred_residuals.weight,nonlinearity='relu')
            nn.init.constant_(self.viewpoint_pred_residuals.bias, 0)
        if self.height_training:
            self.height_pred = nn.Linear(input_size, 2 * num_classes)
            nn.init.xavier_normal_(self.height_pred.weight)
            # torch.nn.init.kaiming_normal_(self.height_pred.weight,nonlinearity='relu')
            nn.init.constant_(self.height_pred.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if self.viewpoint:
            viewpoint_scores = self.viewpoint_pred(x)
            viewpoint_residuals = self.viewpoint_pred_residuals(x) if self.viewpoint_residual else None
            if self.height_training:
                height_scores = self.height_pred(x)
                return scores, proposal_deltas, viewpoint_scores, viewpoint_residuals, height_scores
            return scores, proposal_deltas, viewpoint_scores, viewpoint_residuals, None
        return scores, proposal_deltas, None, None, None
