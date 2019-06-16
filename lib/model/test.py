# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms, soft_nms
from utils.boxTools import *
from utils.dppTools import DPP
import cPickle
from utils.myutils import classToString
from utils.cython_bbox import bbox_overlaps

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import copy

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

classes = ('__background__', 'person', 'bicycle', 'car',
                'motorcycle', 'airplane', 'bus', 'train',
                'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')
NUM_COLORS = len(STANDARD_COLORS)
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  rpn_scores = rois[:, 0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes #, rpn_scores, boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes= im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    # RPN results
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.subplot(aspect='equal')
    # ax.grid(False)
    # ax.axis('off')
    # im_to_out = im[:, :, (2, 1, 0)]
    # im_to_out = np.asarray(im_to_out, dtype="uint8")
    # # patches
    # for kk in range(len(rpn_boxes)):
    #     if rpn_scores[kk]>0.5:
    #           plt.gca().add_patch(
    #             plt.Rectangle((rpn_boxes[kk, 0], rpn_boxes[kk, 1]),
    #                           rpn_boxes[kk, 2] - rpn_boxes[kk, 0],
    #                           rpn_boxes[kk, 3] - rpn_boxes[kk, 1], fill=False,
    #                           edgecolor="DeepPink", linewidth=1)
    #           )
    # plt.imshow(im_to_out)

    # fig.savefig(
    #   '/home/blackfoot/only_eval/tf-faster-rcnn/output/vgg16/rpn_images/' +
    #   '{0:06d}'.format(i) + '.png')
    # plt.close(fig)

    # RCN results

    # scores = scores[:, 1:]
    # M0 = boxes.shape[0]
    # scores_new = copy.deepcopy(scores)
    # num_ignored = scores_new.shape[1] - 5
    # sorted_scores = np.argsort(-scores_new, 1)
    # ignored_cols = np.reshape(sorted_scores[:, -num_ignored:], (M0 * num_ignored))
    # ignored_rows = np.repeat(range(0, sorted_scores.shape[0]), num_ignored)
    # scores_new[ignored_rows, ignored_cols] = 0.0
    # high_scores = np.nonzero(scores >= 0.2)
    # lbl_high_scores = high_scores[1]
    # box_high_scores = high_scores[0]
    # boxes = boxes[:, 4:]
    # input_boxes = np.reshape(
    #   boxes[np.tile(box_high_scores, 4), np.hstack(
    #     (np.multiply(4, lbl_high_scores), np.add(np.multiply(4, lbl_high_scores), 1), \
    #      np.add(np.multiply(4, lbl_high_scores), 2),
    #      np.add(np.multiply(4, lbl_high_scores), 3)))],
    #   (lbl_high_scores.shape[0], 4), order='F')
    # input_clss = np.expand_dims(np.add(lbl_high_scores, 1),-1)
    # input_dets = np.concatenate([input_boxes, input_clss], axis=1)
    #
    # fig = plt.figure(figsize=(14, 7))
    # ax = plt.subplot(aspect='equal')
    # ax.grid(False)
    # ax.axis('off')
    # im_to_out = im[:, :, (2, 1, 0)]
    # im_to_out = np.asarray(im_to_out, dtype="uint8")
    # patches input_dets
    # for iii in range(4):
      # color = STANDARD_COLORS[int(input_dets[iii, 4]) % NUM_COLORS]
      # plt.gca().add_patch(
      #   plt.Rectangle((input_dets[iii, 0], input_dets[iii, 1]),
      #                 input_dets[iii, 2] - input_dets[iii, 0],
      #                 input_dets[iii, 3] - input_dets[iii, 1], fill=False,
      #                 edgecolor=color, linewidth=1)
      # )
    # classes_index = np.unique(input_dets[:, 4]).astype(np.int32)
    # colors = [STANDARD_COLORS[ind % NUM_COLORS] for ind in classes_index]
    # colors = ["Chartreuse", "Purple", 'Orange', 'Tomato']
    # # labels = [str(classToString(classes, ind)) for ind in classes_index]
    # labels = ["cow","sheep","sheep","sheep"]
    # patches = [
    #   mpatches.Patch(color=color, label=label)
    #   for label, color in zip(labels, colors)]
    # plt.gca().legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.60), fancybox=True, shadow=True)
    # plt.imshow(im_to_out)
    #
    # fig.savefig(
    #   '/home/blackfoot/only_eval/tf-faster-rcnn/output/vgg16/rcn_images/' +
    #   '{0:06d}'.format(i) + '.png')
    # plt.close(fig)

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


def dpp_test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.2, vis=False):
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  thresh = cfg.TEST.SCORE_THRESH
  print(
  "===> SCORE_THRESHOLD is: ", thresh)
  # all detections are collected into:
  #    all_boxes[cls][image] = N x 5 array of detections in
  #    (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)

  # timers
  _t = {'im_detect': Timer(), 'misc': Timer()}

  if not cfg.TEST.HAS_RPN:
    roidb = imdb.roidb
  im_dets_pair = {}
  sim_classes = pickle.load(open("/home/blackfoot/only_eval/tf-faster-rcnn/data/coco_semantics.pickle", "r"))

  ff = {}
  for j in xrange(1, imdb.num_classes):
    cls_file = os.path.join(output_dir, '%s.txt' % imdb.classes[j])
    ff[j] = open(cls_file, 'a')

  for i in xrange(num_images):
    # filter out any ground truth boxes
    if cfg.TEST.HAS_RPN:
      box_proposals = None
    else:
      # The roidb may contain ground-truth rois (for example, if the roidb
      # comes from the training or val split). We only want to evaluate
      # detection on the *non*-ground-truth rois. We select those the rois
      # that have the gt_classes field set to 0, which means there's no
      # ground truth.
      box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]
    im = cv2.imread(imdb.image_path_at(i))
    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)

    _t['im_detect'].toc()
    _t['misc'].tic()
    # skip j = 0, because it's the background class
    im_dets_pair[i] = {}
    im_dets_pair[i]['im'] = im
    im_dets_pair[i]['lbl'] = imdb.classes
    score_thresh = thresh
    epsilon = 0.01
    DPP_ = DPP(epsilon=0.02)
    keep = DPP_.dpp_MAP(im_dets_pair[i], scores, boxes, sim_classes, score_thresh, epsilon, max_per_image,
                        close_thr=0.00001)
    if len(keep['box_id']) > 0:
      for j in xrange(1, imdb.num_classes):
        inds = np.where(keep['box_cls'] == j)[0]
        box_ids = keep['box_id'][inds]
        for per_cls in box_ids:
          ff[j].write("%s %g %d %d %d %d\n" % (imdb.image_path_at(i), scores[per_cls, j], boxes[per_cls, 4 * j],
                                               boxes[per_cls, 4 * j + 1], boxes[per_cls, 4 * j + 2],
                                               boxes[per_cls, 4 * j + 3]))
        cls_dets = np.hstack((boxes[box_ids, 4 * j:(j + 1) * 4], scores[box_ids, j][:, np.newaxis])) \
          .astype(np.float32, copy=False)
        all_boxes[j][i] = cls_dets
        im_dets_pair[i][j] = {}
        im_dets_pair[i][j]['dets'] = cls_dets
        # if vis:
        #   vis_detections(im, imdb.classes[j], cls_dets, score_thresh)
    else:
      for j in xrange(1, imdb.num_classes):
        all_boxes[j][i] = np.array([])
    _t['misc'].toc()

    print(
    'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
      .format(i + 1, num_images, _t['im_detect'].average_time,
              _t['misc'].average_time))

  for j in xrange(1, imdb.num_classes):
    ff[j].close()

  det_file = os.path.join(output_dir, 'detections_dpp.pkl')
  with open(det_file, 'wb') as f:
    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

  print(
  'Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


def visualize_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  STANDARD_COLORS = [
    'Chartreuse', 'Aqua', 'Azure', 'MistyRose',
    'HotPink', 'BlueViolet',
    'Cornsilk', 'Cyan', 'SpringGreen', 'Orange','Tomato',  'Yellow',
    'DeepSkyBlue', 'Magenta', 'FloralWhite',
    'ForestGreen', 'Gainsboro', 'Gold','Orange',
     'HotPink','MediumVioletRed','LightSlateGray','Crimson','MintCream','Khaki',
    'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
  ]
  NUM_COLORS = len(STANDARD_COLORS)

  classes = ('__background__', 'person', 'bicycle', 'car',
             'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant',
             'stop sign', 'parking meter', 'bench', 'bird',
             'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag',
             'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard', 'surfboard', 'tennis racket', 'bottle',
             'wine glass', 'cup', 'fork', 'knife',
             'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot',
             'hot dog', 'pizza', 'donut', 'cake',
             'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop',
             'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush')
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  roidb = imdb.roidb

  # num_images
  i = 30
  if True:
  # for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    im = im[100:, 250:, :]

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.grid(False)
    im_to_out = im[:, :, (2, 1, 0)]  # + self.im_mean
    im_to_out = np.asarray(im_to_out, dtype="uint8")
    plt.imshow(im_to_out)

    gt_boxes = roidb[i]['boxes']
    gt_clss = roidb[i]['gt_classes']

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      # inds = np.where(gt_clss == j)[0]
      # cls_dets = gt_boxes[inds,:]

      inds = np.where(scores[:, j] > 0.8)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

      color = STANDARD_COLORS[j % NUM_COLORS]
      for jj in range(len(cls_dets)):
          if cls_dets[jj, 1] <= 2:
              cls_dets[jj, 1] = 2
          if cls_dets[jj, 2] <= 2:
              cls_dets[jj, 2] = 2
          if cls_dets[jj, 2] >= im.shape[1] - 2:
              cls_dets[jj, 2] = cls_dets[jj, 2] - 2
          if cls_dets[jj, 3] >= im.shape[0] - 2:
              cls_dets[jj, 3] = cls_dets[jj, 3] - 2
          plt.gca().add_patch(
              plt.Rectangle((cls_dets[jj, 0], cls_dets[jj, 1]),
                            cls_dets[jj, 2] - cls_dets[jj, 0],
                            cls_dets[jj, 3] - cls_dets[jj, 1], fill=False,
                            edgecolor=color, linewidth=3)
          )
          # if j == 19:
          #     plt.gca().text(
          #         cls_dets[jj, 2] - 55, cls_dets[jj, 1] - 9,
          #         '{:s}'.format(classToString(classes, int(j))),
          #         bbox=dict(facecolor=color, edgecolor=color, alpha=0.7),
          #         fontsize=10, color='black', fontweight='bold')
          # elif j == 4:
          #     plt.gca().text(
          #         cls_dets[jj, 2] - 115, cls_dets[jj, 1] - 9,
          #         '{:s}'.format(classToString(classes, int(j))),
          #         bbox=dict(facecolor=color, edgecolor=color, alpha=0.7),
          #         fontsize=10, color='black', fontweight='bold')
          # elif j == 20:
          #     plt.gca().text(
          #         cls_dets[jj, 0] + 5, cls_dets[jj, 1] + 9,
          #         '{:s}'.format(classToString(classes, int(j))),
          #         bbox=dict(facecolor=color, edgecolor=color, alpha=0.7),
          #         fontsize=10, color='black', fontweight='bold')
          # else:
          #     plt.gca().text(
          #         cls_dets[jj, 0] + 5, cls_dets[jj, 3] - 9,
          #         '{:s}'.format(classToString(classes, int(j))),
          #         bbox=dict(facecolor=color, edgecolor=color, alpha=0.7),
          #         fontsize=10, color='black', fontweight='bold')
          plt.gca().text(
              cls_dets[jj, 0]+3, cls_dets[jj, 1]-6,
              '{:s}'.format(classToString(classes, int(j))),
              bbox=dict(facecolor=color, edgecolor=color, alpha=1),
              fontsize=10, color='black', fontweight='bold')


    fig.savefig(
      '/home/blackfoot/only_eval/tf-faster-rcnn/output/vgg16/coco_2014_train/orig_output/' + str(i))
    plt.close(fig)

    print(
    'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
      .format(i + 1, num_images, _t['im_detect'].average_time,
              _t['misc'].average_time))

def lddp_visualize_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  sim_classes = pickle.load(open("/home/blackfoot/only_eval/lddp-tf-faster-rcnn/data/coco_semantics.pickle", "r"))
  STANDARD_COLORS = [
    'Chartreuse', 'Aqua', 'Azure', 'MistyRose',
    'HotPink', 'BlueViolet',
    'Cornsilk', 'Cyan', 'SpringGreen', 'Orange','Tomato',  'Yellow',
    'DeepSkyBlue', 'Magenta', 'FloralWhite',
    'ForestGreen', 'Gainsboro', 'Gold','Orange',
     'HotPink','MediumVioletRed','LightSlateGray','Crimson','MintCream','Khaki',
    'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
  ]
  NUM_COLORS = len(STANDARD_COLORS)

  classes = ('__background__', 'person', 'bicycle', 'car',
             'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant',
             'stop sign', 'parking meter', 'bench', 'bird',
             'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag',
             'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard', 'surfboard', 'tennis racket', 'bottle',
             'wine glass', 'cup', 'fork', 'knife',
             'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot',
             'hot dog', 'pizza', 'donut', 'cake',
             'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop',
             'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush')

  im_dets_pair = {}

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  # num_images
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.grid(False)
    im_to_out = im[:, :, (2, 1, 0)]  # + self.im_mean
    im_to_out = np.asarray(im_to_out, dtype="uint8")
    plt.imshow(im_to_out)
    im_dets_pair[i] = {}
    im_dets_pair[i]['im'] = im
    im_dets_pair[i]['lbl'] = imdb.classes

    score_thresh = 0.6
    epsilon = 0.01
    DPP_ = DPP(epsilon=0.02)
    keep = DPP_.dpp_MAP(im_dets_pair[i], scores, boxes, sim_classes, score_thresh, epsilon, max_per_image,
                        close_thr=0.00001)
    # skip j = 0, because it's the background class
    if len(keep['box_id']) > 0:
      for j in xrange(1, imdb.num_classes):
        inds = np.where(keep['box_cls'] == j)[0]
        box_ids = keep['box_id'][inds]
        cls_dets = np.hstack((boxes[box_ids, 4 * j:(j + 1) * 4], scores[box_ids, j][:, np.newaxis])) \
          .astype(np.float32, copy=False)
        inds = np.where(scores[:, j] > score_thresh)[0]
        cls_boxes = boxes[inds, j*4:(j+1)*4]

        color = STANDARD_COLORS[j % NUM_COLORS]
        for jj in range(len(cls_dets)):
            if cls_dets[jj, 1] <= 2:
                cls_dets[jj, 1] = 2
            if cls_dets[jj, 2] <= 2:
                cls_dets[jj, 2] = 2
            if cls_dets[jj, 2] >= im.shape[1] - 2:
                cls_dets[jj, 2] = cls_dets[jj, 2] - 2
            if cls_dets[jj, 3] >= im.shape[0] - 2:
                cls_dets[jj, 3] = cls_dets[jj, 3] - 2
            plt.gca().add_patch(
                plt.Rectangle((cls_dets[jj, 0], cls_dets[jj, 1]),
                              cls_dets[jj, 2] - cls_dets[jj, 0],
                              cls_dets[jj, 3] - cls_dets[jj, 1], fill=False,
                              edgecolor=color, linewidth=3)
            )
            plt.gca().text(
                cls_dets[jj, 0] + 3, cls_dets[jj, 1] - 6,
                '{:s}'.format(classToString(classes, int(j))),
                bbox=dict(facecolor=color, edgecolor=color),
                fontsize=10, color='black', fontweight='bold')

    fig.savefig(
      '/home/blackfoot/only_eval/tf-faster-rcnn/output/vgg16/coco_2014_train/lddp_output2/' + str(i))
    plt.close(fig)

    print(
    'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
      .format(i + 1, num_images, _t['im_detect'].average_time,
              _t['misc'].average_time))


def test_nms_overlap(sess, net, imdb, overlap_thresh):
  print ("nms")
  print (overlap_thresh)
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  roidb = imdb.roidb

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  # import copy
  obj_all = np.zeros(5)
  hist0 = np.zeros(imdb.num_classes)
  hist_all0 = np.zeros(imdb.num_classes)
  hist1 = np.zeros(imdb.num_classes)
  hist_all1 = np.zeros(imdb.num_classes)
  hist2 = np.zeros(imdb.num_classes)
  hist_all2 = np.zeros(imdb.num_classes)
  hist3 = np.zeros(imdb.num_classes)
  hist_all3 = np.zeros(imdb.num_classes)
  hist4 = np.zeros(imdb.num_classes)
  hist_all4 = np.zeros(imdb.num_classes)
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    gt_boxes = roidb[i]['boxes']
    gt_clss = roidb[i]['gt_classes']

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()
    thresh = 0.0
    # overlap_thresh = 0.4
    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      objects0 = []
      objects1 = []
      objects2 = []
      objects3 = []
      objects4 = []
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      cls_gt_boxes = gt_boxes[gt_clss == j]

      overlaps_temp = bbox_overlaps(
          np.ascontiguousarray(cls_gt_boxes, dtype=np.float),
          np.ascontiguousarray(cls_gt_boxes, dtype=np.float))
      index0 = []
      index1 = []
      index2 = []
      index3 = []
      index4 = []
      if len(overlaps_temp) > 1:
          for ii in range(len(cls_gt_boxes) - 1):
              for jj in range(ii + 1, len(cls_gt_boxes)):
                  if overlaps_temp[ii, jj] > 0.0:
                      if ii not in index0:
                          objects0.append(cls_gt_boxes[ii])
                          index0 = np.append(index0, int(ii))
                      if jj not in index0:
                          objects0.append(cls_gt_boxes[jj])
                          index0 = np.append(index0, int(jj))
                  if overlaps_temp[ii, jj] > 0.1:
                      if ii not in index1:
                          objects1.append(cls_gt_boxes[ii])
                          index1 = np.append(index1, int(ii))
                      if jj not in index1:
                          objects1.append(cls_gt_boxes[jj])
                          index1 = np.append(index1, int(jj))
                  if overlaps_temp[ii, jj] > 0.2:
                      if ii not in index2:
                          objects2.append(cls_gt_boxes[ii])
                          index2 = np.append(index2, int(ii))
                      if jj not in index2:
                          objects2.append(cls_gt_boxes[jj])
                          index2 = np.append(index2, int(jj))
                  if overlaps_temp[ii, jj] > 0.3:
                      if ii not in index3:
                          objects3.append(cls_gt_boxes[ii])
                          index3 = np.append(index3, int(ii))
                      if jj not in index3:
                          objects3.append(cls_gt_boxes[jj])
                          index3 = np.append(index3, int(jj))
                  if overlaps_temp[ii, jj] > 0.4:
                      if ii not in index4:
                          objects4.append(cls_gt_boxes[ii])
                          index4 = np.append(index4, int(ii))
                      if jj not in index4:
                          objects4.append(cls_gt_boxes[jj])
                          index4 = np.append(index4, int(jj))
      if len(objects0) > 0:
          overlaps = bbox_overlaps(
              np.ascontiguousarray(cls_dets, dtype=np.float),
              np.ascontiguousarray(objects0, dtype=np.float))
          assign_gt_ind = np.argmax(overlaps, 1)
          detected0 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
          obj_all[0] += len(cls_gt_boxes)
          hist_all0[j] += len(objects0)
          hist0[j] += len(detected0)
      if len(objects1) > 0:
          overlaps = bbox_overlaps(
              np.ascontiguousarray(cls_dets, dtype=np.float),
              np.ascontiguousarray(objects1, dtype=np.float))
          assign_gt_ind = np.argmax(overlaps, 1)
          detected1 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
          obj_all[1] += len(cls_gt_boxes)
          hist_all1[j] += len(objects1)
          hist1[j] += len(detected1)
      if len(objects2) > 0:
          overlaps = bbox_overlaps(
              np.ascontiguousarray(cls_dets, dtype=np.float),
              np.ascontiguousarray(objects2, dtype=np.float))
          assign_gt_ind = np.argmax(overlaps, 1)
          detected2 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
          obj_all[2] += len(cls_gt_boxes)
          hist_all2[j] += len(objects2)
          hist2[j] += len(detected2)
      if len(objects3) > 0:
          overlaps = bbox_overlaps(
              np.ascontiguousarray(cls_dets, dtype=np.float),
              np.ascontiguousarray(objects3, dtype=np.float))
          assign_gt_ind = np.argmax(overlaps, 1)
          detected3 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
          obj_all[3] += len(cls_gt_boxes)
          hist_all3[j] += len(objects3)
          hist3[j] += len(detected3)
      if len(objects4) > 0:
          overlaps = bbox_overlaps(
              np.ascontiguousarray(cls_dets, dtype=np.float),
              np.ascontiguousarray(objects4, dtype=np.float))
          assign_gt_ind = np.argmax(overlaps, 1)
          detected4 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
          obj_all[4] += len(cls_gt_boxes)
          hist_all4[j] += len(objects4)
          hist4[j] += len(detected4)

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time,
                  _t['misc'].average_time))
  print(float(sum(hist0)) / float(sum(hist_all0)))
  print(sum(hist_all0))
  print(float(sum(hist1)) / float(sum(hist_all1)))
  print(sum(hist_all1))
  print(float(sum(hist2)) / float(sum(hist_all2)))
  print(sum(hist_all2))
  print(float(sum(hist3)) / float(sum(hist_all3)))
  print(sum(hist_all3))
  print(float(sum(hist4)) / float(sum(hist_all4)))
  print(sum(hist_all4))
  print(obj_all)


def test_ldpp_overlap(sess, net, imdb, overlap_thresh):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  roidb = imdb.roidb
  sim_classes = pickle.load(open("/home/blackfoot/only_eval/tf-faster-rcnn/data/coco_semantics.pickle", "r"))

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  import copy
  obj_all = np.zeros(5)
  hist0 = np.zeros(imdb.num_classes)
  hist_all0 = np.zeros(imdb.num_classes)
  hist1 = np.zeros(imdb.num_classes)
  hist_all1 = np.zeros(imdb.num_classes)
  hist2 = np.zeros(imdb.num_classes)
  hist_all2 = np.zeros(imdb.num_classes)
  hist3 = np.zeros(imdb.num_classes)
  hist_all3 = np.zeros(imdb.num_classes)
  hist4 = np.zeros(imdb.num_classes)
  hist_all4 = np.zeros(imdb.num_classes)
  im_dets_pair = {}
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    gt_boxes = roidb[i]['boxes']
    gt_clss = roidb[i]['gt_classes']

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()
    im_dets_pair[i] = {}
    im_dets_pair[i]['im'] = im
    im_dets_pair[i]['lbl'] = imdb.classes

    _t['misc'].tic()
    score_thresh = 0.05
    # skip j = 0, because it's the background class
    epsilon = 0.01
    DPP_ = DPP(epsilon=0.02)
    keep = DPP_.dpp_MAP(im_dets_pair[i], scores, boxes, sim_classes, score_thresh, epsilon, 100,
                        close_thr=0.00001)
    if len(keep['box_id']) > 0:
      for j in xrange(1, imdb.num_classes):
        objects0 = []
        objects1 = []
        objects2 = []
        objects3 = []
        objects4 = []
        inds = np.where(keep['box_cls'] == j)[0]
        box_ids = keep['box_id'][inds]
        cls_dets = np.hstack((boxes[box_ids, 4 * j:(j + 1) * 4], scores[box_ids, j][:, np.newaxis])) \
          .astype(np.float32, copy=False)
        cls_gt_boxes = gt_boxes[gt_clss == j]

        overlaps_temp = bbox_overlaps(
          np.ascontiguousarray(cls_gt_boxes, dtype=np.float),
          np.ascontiguousarray(cls_gt_boxes, dtype=np.float))
        index0 = []
        index1 = []
        index2 = []
        index3 = []
        index4 = []
        if len(overlaps_temp) > 1:
            for ii in range(len(cls_gt_boxes) - 1):
                for jj in range(ii + 1, len(cls_gt_boxes)):
                    if overlaps_temp[ii, jj] > 0.0:
                        if ii not in index0:
                            objects0.append(cls_gt_boxes[ii])
                            index0 = np.append(index0, int(ii))
                        if jj not in index0:
                            objects0.append(cls_gt_boxes[jj])
                            index0 = np.append(index0, int(jj))
                    if overlaps_temp[ii, jj] > 0.1:
                        if ii not in index1:
                            objects1.append(cls_gt_boxes[ii])
                            index1 = np.append(index1, int(ii))
                        if jj not in index1:
                            objects1.append(cls_gt_boxes[jj])
                            index1 = np.append(index1, int(jj))
                    if overlaps_temp[ii, jj] > 0.2:
                        if ii not in index2:
                            objects2.append(cls_gt_boxes[ii])
                            index2 = np.append(index2, int(ii))
                        if jj not in index2:
                            objects2.append(cls_gt_boxes[jj])
                            index2 = np.append(index2, int(jj))
                    if overlaps_temp[ii, jj] > 0.3:
                        if ii not in index3:
                            objects3.append(cls_gt_boxes[ii])
                            index3 = np.append(index3, int(ii))
                        if jj not in index3:
                            objects3.append(cls_gt_boxes[jj])
                            index3 = np.append(index3, int(jj))
                    if overlaps_temp[ii, jj] > 0.4:
                        if ii not in index4:
                            objects4.append(cls_gt_boxes[ii])
                            index4 = np.append(index4, int(ii))
                        if jj not in index4:
                            objects4.append(cls_gt_boxes[jj])
                            index4 = np.append(index4, int(jj))
        if len(objects0) > 0:
            overlaps = bbox_overlaps(
                np.ascontiguousarray(cls_dets, dtype=np.float),
                np.ascontiguousarray(objects0, dtype=np.float))
            assign_gt_ind = np.argmax(overlaps, 1)
            detected0 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
            obj_all[0] += len(cls_gt_boxes)
            hist_all0[j] += len(objects0)
            hist0[j] += len(detected0)
        if len(objects1) > 0:
            overlaps = bbox_overlaps(
                np.ascontiguousarray(cls_dets, dtype=np.float),
                np.ascontiguousarray(objects1, dtype=np.float))
            assign_gt_ind = np.argmax(overlaps, 1)
            detected1 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
            obj_all[1] += len(cls_gt_boxes)
            hist_all1[j] += len(objects1)
            hist1[j] += len(detected1)
        if len(objects2) > 0:
            overlaps = bbox_overlaps(
                np.ascontiguousarray(cls_dets, dtype=np.float),
                np.ascontiguousarray(objects2, dtype=np.float))
            assign_gt_ind = np.argmax(overlaps, 1)
            detected2 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
            obj_all[2] += len(cls_gt_boxes)
            hist_all2[j] += len(objects2)
            hist2[j] += len(detected2)
        if len(objects3) > 0:
            overlaps = bbox_overlaps(
                np.ascontiguousarray(cls_dets, dtype=np.float),
                np.ascontiguousarray(objects3, dtype=np.float))
            assign_gt_ind = np.argmax(overlaps, 1)
            detected3 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
            obj_all[3] += len(cls_gt_boxes)
            hist_all3[j] += len(objects3)
            hist3[j] += len(detected3)
        if len(objects4) > 0:
            overlaps = bbox_overlaps(
                np.ascontiguousarray(cls_dets, dtype=np.float),
                np.ascontiguousarray(objects4, dtype=np.float))
            assign_gt_ind = np.argmax(overlaps, 1)
            detected4 = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
            obj_all[4] += len(cls_gt_boxes)
            hist_all4[j] += len(objects4)
            hist4[j] += len(detected4)

      print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['misc'].average_time))
  print(float(sum(hist0)) / float(sum(hist_all0)))
  print(sum(hist_all0))
  print(float(sum(hist1)) / float(sum(hist_all1)))
  print(sum(hist_all1))
  print(float(sum(hist2)) / float(sum(hist_all2)))
  print(sum(hist_all2))
  print(float(sum(hist3)) / float(sum(hist_all3)))
  print(sum(hist_all3))
  print(float(sum(hist4)) / float(sum(hist_all4)))
  print(sum(hist_all4))
  print(obj_all)


def test_oracle(sess, net, imdb, overlap_thresh):
  print("nms")
  print(overlap_thresh)
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  roidb = imdb.roidb

  # timers
  _t = {'im_detect': Timer(), 'misc': Timer()}
  # import copy
  hist = np.zeros(imdb.num_classes)
  hist_all = np.zeros(imdb.num_classes)
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    gt_boxes = roidb[i]['boxes']
    gt_clss = roidb[i]['gt_classes']

    _t['im_detect'].tic()
    # scores, boxes = im_detect(sess, net, im)

    boxes = gt_boxes.astype(np.float32,copy=False)
    scores = np.zeros((len(gt_boxes),81))
    for obj in range(len(gt_boxes)):
      scores[obj][gt_clss[obj]] = 1
    boxes = np.repeat(scores,4,axis=1) * np.repeat(boxes,81,axis=0).reshape(-1,4*81)

    _t['im_detect'].toc()

    _t['misc'].tic()
    thresh = 0.0
    # overlap_thresh = 0.4
    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      objects = []
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = soft_nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      cls_gt_boxes = gt_boxes[gt_clss == j]

      overlaps_temp = bbox_overlaps(
        np.ascontiguousarray(cls_gt_boxes, dtype=np.float),
        np.ascontiguousarray(cls_gt_boxes, dtype=np.float))
      index = []
      if len(overlaps_temp) > 1:
        for ii in range(len(cls_gt_boxes) - 1):
          for jj in range(ii + 1, len(cls_gt_boxes)):
            if overlaps_temp[ii, jj] > overlap_thresh:
              if ii not in index:
                objects.append(cls_gt_boxes[ii])
                index = np.append(index, int(ii))
              if jj not in index:
                objects.append(cls_gt_boxes[jj])
                index = np.append(index, int(jj))
      if len(objects) > 0:
        overlaps = bbox_overlaps(
          np.ascontiguousarray(cls_dets, dtype=np.float),
          np.ascontiguousarray(objects, dtype=np.float))
        assign_gt_ind = np.argmax(overlaps, 1)
        detected = np.unique(assign_gt_ind[np.max(overlaps, 1) > 0.5000])
        hist_all[j] += len(objects)
        hist[j] += len(detected)
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time,
                  _t['misc'].average_time))
  print (float(sum(hist))/float(sum(hist_all)))
  print(hist_all)



def visualize_score(sess, net, imdb, weights_filename, max_per_image=100, thresh=-0.5, sim_thresh=0.55):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)

  classes = ('__background__', 'person', 'bicycle', 'car',
             'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant',
             'stop sign', 'parking meter', 'bench', 'bird',
             'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag',
             'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard', 'surfboard', 'tennis racket', 'bottle',
             'wine glass', 'cup', 'fork', 'knife',
             'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot',
             'hot dog', 'pizza', 'donut', 'cake',
             'chair', 'couch', 'potted plant', 'bed',
             'dining table', 'toilet', 'tv', 'laptop',
             'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush')

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  # num_images
  STANDARD_COLORS = [
    'LightPink', 'Ivory', 'Yellow', 'Silver',
    'Lime', 'LightCoral',
    'LightSalmon', 'LightSlateGray', 'SpringGreen', 'Orange','Tomato',  'Yellow',
    'DeepSkyBlue', 'Magenta', 'FloralWhite',
    'ForestGreen', 'Gainsboro', 'Gold','DeepSkyBlue',
     'Orange','MediumVioletRed','LightSlateGray','Crimson','MintCream','Khaki',
    'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
  ]
  NUM_COLORS = len(STANDARD_COLORS)
  i = 1018
  # for i in range(num_images):
  if True:
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    ax.grid(False)
    im_to_out = im[:, :, (2, 1, 0)]  # + self.im_mean
    im_to_out = np.asarray(im_to_out, dtype="uint8")
    plt.imshow(im_to_out)
    pred_clss = []
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > 0.1)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      max_score = 0.0
      if len(inds)>0:
          pred_clss.append(j)
      for jj in range(len(cls_dets)):
        color = STANDARD_COLORS[j % NUM_COLORS]
        if cls_dets[jj, 1] <= 2:
          cls_dets[jj, 1] = 2
        if cls_dets[jj, 2] <= 2:
          cls_dets[jj, 2] = 2
        if cls_dets[jj, 2] >= im.shape[1]-2:
         cls_dets[jj, 2] = cls_dets[jj, 2] - 2
        if cls_dets[jj, 3] >= im.shape[0]-2:
          cls_dets[jj, 3] = cls_dets[jj, 3] - 2
        plt.gca().add_patch(
          plt.Rectangle((cls_dets[jj, 0], cls_dets[jj, 1]),
                       cls_dets[jj, 2] - cls_dets[jj, 0],
                       cls_dets[jj, 3] - cls_dets[jj, 1], fill=False,
                       edgecolor=color, linewidth=0.4))
        if max_score<cls_dets[jj,-1]:
          max_jj = jj
          max_score = cls_dets[jj,-1]
      if len(cls_dets)>0:
        if j == 19:
          plt.gca().text(
            cls_dets[max_jj, 2] - 43, cls_dets[max_jj, 1] + 16,
            '{0:.5f}'.format(cls_dets[max_jj, -1]),
            bbox=dict(facecolor=color, edgecolor=color, alpha=0.5),
            fontsize=10, color='black', fontweight='bold')
        elif j == 18:
          plt.gca().text(
            cls_dets[max_jj, 0] + 3, cls_dets[max_jj, 3] - 6,
            '{0:.5f}'.format(cls_dets[max_jj, -1]),
            bbox=dict(facecolor=color, edgecolor=color, alpha=0.5),
            fontsize=10, color='black', fontweight='bold')
        else:
          plt.gca().text(
            cls_dets[max_jj, 0]+3, cls_dets[max_jj, 1]+16,
            '{0:.5f}'.format(cls_dets[max_jj,-1]),
            bbox=dict(facecolor=color, edgecolor=color,alpha=0.8),
            fontsize=10, color='black',fontweight='bold')
    classes_index = np.unique(pred_clss).astype(np.int32)
    labels = [str(classToString(classes, ind)) for ind in classes_index]
    colors = [STANDARD_COLORS[ind % NUM_COLORS] for ind in classes_index]
    patches = [
      mpatches.Patch(color=color, label=label)
      for label, color in zip(labels, colors)]
    plt.gca().legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.60), fancybox=True, shadow=True)

  fig.savefig('/home/blackfoot/only_eval/tf-faster-rcnn/output/vgg16/coco_2014_train/frcnn_score/'  + str(i))
  plt.close(fig)

  print(
  'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
    .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

