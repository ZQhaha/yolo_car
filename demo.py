from PIL import ImageDraw, ImageFont, Image

from plateNet import myNet_ocr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time

plate_chr = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"


def cv_imread(path):  # 读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds


def image_processing(img, device, img_size):
    img_h, img_w = img_size
    img = cv2.resize(img, (img_w, img_h))
    # img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, img_size):
    # img = cv2.imread(image_path)
    input = image_processing(img, device, img_size)
    preds = model(input)
    preds = preds.argmax(dim=2)
    # print(preds)
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate


def init_model(device, model_path):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)  # export  True 用来推理
    # model =build_lprnet(num_classes=len(plate_chr),export=True)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def recognition(model_path, img, img_h=48, img_w=168):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (img_h, img_w)
    model = init_model(device, model_path)
    plate = get_plate_result(img, device, model, img_size)
    return plate
