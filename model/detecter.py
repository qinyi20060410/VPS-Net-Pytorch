from __future__ import division

from net.yolo import *
from util.utils import *
from util.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder",
                        type=str,
                        default="data/training",
                        help="path to dataset")
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path",
                        type=str,
                        default="checkpoints/yolov3_4.pth",
                        help="path to weights file")
    parser.add_argument("--class_path",
                        type=str,
                        default="data/angle_type.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres",
                        type=float,
                        default=0.8,
                        help="object confidence threshold")
    parser.add_argument("--nms_thres",
                        type=float,
                        default=0.4,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="size of the batches")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=0,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    parser.add_argument("--checkpoint_model",
                        type=str,
                        help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available(
    ) else torch.FloatTensor

    imgs = []  # Stores image paths
    marks = []
    slots = []
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, slot, mark) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres,
                                             opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        slots.append(slot)
        marks.append(mark)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections,
                mark) in enumerate(zip(imgs, img_detections, marks)):

        print("(%d) Image: '%s'" % (img_i, path))

        # print(slot)
        # print(slot[0])
        print(mark)

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # for crop
        num_boxes = 0
        points_x = []
        points_y = []
        ps = []

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, (600, 600))
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" %
                      (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(
                    np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1),
                                         box_w,
                                         box_h,
                                         linewidth=2,
                                         edgecolor=color,
                                         facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={
                        "color": color,
                        "pad": 0
                    },
                )

                # for crop
                if int(cls_pred) == 3:
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    points_x.append(x)
                    points_y.append(y)

                else:
                    num_boxes += 1

            # for crop
            if num_boxes > 0:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
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
                            point1, point2 = from_head_points(bbx, img)
                        if int(cls_pred) == 0:
                            angle = 90
                        elif int(cls_pred) == 1:
                            angle = 67
                        elif int(cls_pred) == 2:
                            angle = 129
                        ps.append([point1, point2, angle])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(len(ps))
        if len(ps) != 0:
            index = 0
            for p in ps:
                point1 = p[0]
                point2 = p[1]
                angle = p[2]

                cv2.circle(img,
                           (round(float(point1[0])), round(float(point1[1]))),
                           10, (0, 0, 255))

                cv2.circle(img,
                           (round(float(point2[0])), round(float(point2[1]))),
                           10, (0, 0, 255))

                pts = compute_four_points(angle, point1, point2)
                point3_org = copy.copy(pts[2])
                point4_org = copy.copy(pts[3])

                regul_img = image_preprocess(img, pts)

                # filename = path.split("/")[-1].split(".")[0] + '_' + str(index)
                # cv2.imwrite(f"output/generate/{filename}.jpg", regul_img)
                # cv2.imshow('crop', regul_img)
                index = index + 1

        # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png",
        #             bbox_inches="tight",
        #             pad_inches=0.0)
        # plt.close()

        cv2.imshow('img', img[:, :, ::-1])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(0) & 0xFF == ord('n'):
            continue

    cv2.destroyAllWindows()