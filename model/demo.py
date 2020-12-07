from __future__ import division

import argparse
import torch
import os
import cv2
import numpy as np
import copy
import glob
import tqdm
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import cv2

from net.yolo import Darknet
from net.classify_net import Classify_Net
from util.utils import non_max_suppression, rescale_boxes, pad_to_square, resize, crop_margin, fixed_ROI, compute_four_points


class PsDetect(object):
    """
    Return paired marking points and angle
    """
    def __init__(self, model_def, model_path, img_size, device):
        self.model_yolov3 = Darknet(model_def, img_size=img_size).to(device)
        self.model_yolov3.load_state_dict(torch.load(model_path))
        self.model_yolov3.eval()
        self.device = device
        self.img_size = img_size

    # Detect the head of the parking slot and marking points
    def detect_object(self, img, conf_thres, nms_thres):
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        detection = self.model_yolov3(img)
        # From [x,y,w,h] to [x_l,y_l,x_r,y_r]
        detection = non_max_suppression(detection, conf_thres, nms_thres)
        if detection[0] is not None:
            detection = rescale_boxes(detection[0], self.img_size, (600, 600))
        return detection

    # The points in the head of the parking slot is less than 2
    def from_head_points(self, bbx, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1_l = bbx[0] + 24
        y1_l = bbx[1] + 22
        x2_l = bbx[2] - 24
        y2_l = bbx[3] - 22
        x1_r = bbx[2] - 24
        y1_r = bbx[1] + 22
        x2_r = bbx[0] + 24
        y2_r = bbx[3] - 22
        k_l = (y2_l - y1_l) / (x2_l - x1_l)
        k_r = (y1_r - y2_r) / (x1_r - x2_r)
        sum_intensity_l = 0
        sum_intensity_r = 0
        for i in range(int(x2_l - x1_l)):
            for k in range(-2, 2):
                y = int(k_l * i + y1_l + k)
                x = int(i + x1_l)
                if y > 599:
                    y = 599
                if x > 599:
                    x = 599
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0
                sum_intensity_l += gray_img[y, x]
        for i in range(int(x1_r - x2_r)):
            for k in range(-2, 2):
                y = int(k_r * i + y2_r + k)
                x = int(i + x2_r)
                if y > 599:
                    y = 599
                if x > 599:
                    x = 599
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0
                sum_intensity_r += gray_img[y, x]
        if sum_intensity_l > sum_intensity_r:
            if y2_l > y1_l:
                point1_x = x1_l
                point1_y = y1_l
                point2_x = x2_l
                point2_y = y2_l
            else:
                point1_x = x2_l
                point1_y = y2_l
                point2_x = x1_l
                point2_y = y1_l
        else:
            if y2_r > y1_r:
                point1_x = x1_r
                point1_y = y1_r
                point2_x = x2_r
                point2_y = y2_r
            else:
                point1_x = x2_r
                point1_y = y2_r
                point2_x = x1_r
                point2_y = y1_r
        point1 = np.array([point1_x, point1_y])
        point2 = np.array([point2_x, point2_y])
        return point1, point2

    # Parking slot detection
    def detect_ps(self, img, conf_thres, nms_thres):
        detection = self.detect_object(img, conf_thres, nms_thres)
        num_boxes = 0
        points_x = []
        points_y = []
        ps = []
        if detection[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                if int(cls_pred) == 3:
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    points_x.append(x)
                    points_y.append(y)
                else:
                    num_boxes += 1
            if num_boxes > 0:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    points_valid_x = []
                    points_valid_y = []
                    if int(cls_pred != 3):
                        for x, y in zip(points_x, points_y):
                            if x > x1 and x < x2 and y > y1 and y < y2:
                                points_valid_x.append(x)
                                points_valid_y.append(y)
                        if len(points_valid_x) == 2:
                            point1 = np.array((points_valid_x[0].item(),
                                               points_valid_y[0].item()))
                            point2 = np.array((points_valid_x[1].item(),
                                               points_valid_y[1].item()))
                        else:
                            bbx = [x1, y1, x2, y2]
                            point1, point2 = self.from_head_points(bbx, img)
                        if int(cls_pred) == 0:
                            angle = 90
                        elif int(cls_pred) == 1:
                            angle = 67
                        else:
                            angle = 129
                        ps.append([point1, point2, angle])
        return ps


class vpsClassify(object):
    """
    Return whether the paking slot is vacant. 0: vacant 1: non-vancant
    """
    def __init__(self, model_path, device):
        self.model_customized = Classify_Net()
        self.model_customized.eval()
        self.model_customized.load_state_dict(torch.load(model_path))
        self.model_customized.to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.device = device

    # parking slot regularization
    def image_preprocess(self, img, pts):
        points_roi = crop_margin(pts[0], pts[1], pts[2], pts[3])
        roi_img = fixed_ROI(img, points_roi)
        crop_x_min = np.min(points_roi[:, 0]) + 1
        crop_x_max = np.max(points_roi[:, 0])
        crop_y_min = np.min(points_roi[:, 1]) + 1
        crop_y_max = np.max(points_roi[:, 1])
        if pts[1][1] > pts[0][1]:
            points_dst = np.array(
                [[crop_x_max, crop_y_min], [crop_x_max, crop_y_max],
                 [crop_x_min, crop_y_max], [crop_x_min, crop_y_min]],
                np.float32)
        else:
            points_dst = np.array(
                [[crop_x_max, crop_y_min], [crop_x_max, crop_y_max],
                 [crop_x_min, crop_y_max], [crop_x_min, crop_y_min]],
                np.float32)
        m_warp = cv2.getPerspectiveTransform(points_roi, points_dst)
        warp_img = cv2.warpPerspective(roi_img, m_warp, (600, 600))
        crop_img = warp_img[int(crop_y_min):int(crop_y_max),
                            int(crop_x_min):int(crop_x_max)]
        if (crop_img.shape[0] / crop_img.shape[1]) > 2:
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
        regul_img = cv2.resize(crop_img, (120, 46))
        return regul_img

    # parking slot occupancy classification
    def vps_classify(self, img, pts):
        regul_img = self.image_preprocess(img, pts)
        regul_img = self.transform(regul_img)
        regul_img = regul_img.to(self.device)
        regul_img = regul_img.unsqueeze(0)
        output = self.model_customized(regul_img)
        _, pred = torch.max(output.data, 1)
        return pred.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder",
                        type=str,
                        default="data/outdoor-rainy",
                        help="path to dataset")
    parser.add_argument("--output_folder",
                        type=str,
                        default="output/outdoor-rainy",
                        help="path to output")
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path_yolo",
                        type=str,
                        default="weights/yolov3_4.pth",
                        help="path to yolo weights file")
    parser.add_argument("--weights_path_vps",
                        type=str,
                        default="checkpoints/classify_ckpt_best.pth",
                        help="path to vps weights file")
    parser.add_argument("--conf_thres",
                        type=float,
                        default=0.9,
                        help="object confidence threshold")  # 0.9
    parser.add_argument("--nms_thres",
                        type=float,
                        default=0.5,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    parser.add_argument("--save_files",
                        type=bool,
                        default=False,
                        help="save detected results")
    opt = parser.parse_args()

    os.makedirs(opt.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps_detect = PsDetect(opt.model_def, opt.weights_path_yolo, opt.img_size,
                         device)
    vps_classify = vpsClassify(opt.weights_path_vps, device)

    with torch.no_grad():
        imgs_list = glob.glob(opt.input_folder + '/*.jpg')
        print(opt.input_folder)
        print(len(imgs_list))
        for img_path in tqdm.tqdm(imgs_list):
            if opt.save_files:
                img_name = img_path.split('/')[-1]
                filename = img_name.split('.')[0] + '.txt'
                file_path = os.path.join(opt.output_folder, filename)
                file = open(file_path, 'w')
            img = np.array(Image.open(img_path))
            if img.shape[0] != 600:
                img = cv2.resize(img, (600, 600))
            detections = ps_detect.detect_ps(img, opt.conf_thres,
                                             opt.nms_thres)
            if len(detections) != 0:
                for detection in detections:
                    point1 = detection[0]
                    point2 = detection[1]
                    angle = detection[2]
                    pts = compute_four_points(angle, point1, point2)
                    point3_org = copy.copy(pts[2])
                    point4_org = copy.copy(pts[3])
                    label_vacant = vps_classify.vps_classify(img, pts)
                    if label_vacant == 0:
                        color = (0, 255, 0)
                    elif label_vacant == 1:
                        color = (255, 0, 0)
                    elif label_vacant == 2:
                        color = (0, 0, 255)
                    pts_show = np.array(
                        [pts[0], pts[1], point3_org, point4_org], np.int32)
                    if opt.save_files:
                        file.write(str(angle))
                        file.write(' ')
                        points = list(
                            (pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
                        for value in points:
                            file.write(str(value.item()))
                            file.write(' ')
                        file.write('\n')
                    cv2.polylines(img, [pts_show], True, color, 2)
            cv2.imshow('Detect PS', img[:, :, ::-1])
            cv2.waitKey(1000)
            if opt.save_files:
                file.close()
                cv2.imwrite(os.path.join(opt.output_folder, img_name),
                            img[:, :, ::-1])
        cv2.destroyAllWindows()