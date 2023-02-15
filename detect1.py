import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import demo
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes


def read_and_resize(img_path):
    img_ori = cv2.imread(img_path)
    shape = img_ori.shape[:2]
    # 计算resize ratio和 padding
    r = min(640 / shape[0], 640 / shape[1])  # 640:new_shape
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    dw /= 2
    dh /= 2
    # 执行resize和padding
    if shape[::-1] != new_unpad:  # resize
        im_new = cv2.resize(img_ori, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_new = cv2.copyMakeBorder(im_new, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

    return img_ori, img_new, ratio, (dw, dh)


def text_border(draw, xyxy, text, font, shadowcolor, fillcolor):
    label_size1 = font.getsize(text)
    text_origin1 = np.array([xyxy[0], xyxy[1] - 2 * label_size1[1]])
    rec = draw.textbbox(text_origin1, text, font=font)
    draw.rectangle(rec, outline='red', width=3, fill='red')
    x, y = text_origin1
    # thin border
    draw.text((x - 1, y), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y), text, font=font, fill=shadowcolor)
    draw.text((x, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x, y + 1), text, font=font, fill=shadowcolor)

    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)

    # now draw the text over it
    draw.text((x, y), text, font=font, fill=fillcolor)


def main(yolo_path, text_path, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(yolo_path, device=device, dnn=False, data="data/ccpd_green.yaml", fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgz = [640, 640]
    img_ori, img_new = read_and_resize(img_path)[:2]
    img_new = img_new.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_new = np.ascontiguousarray(img_new)
    # todo: no_warmup, Profile()

    img_new = torch.from_numpy(img_new).to(device)
    img_new = img_new.float()  # uint8 to fp16/32
    img_new /= 255  # 0 - 255 to 0.0 - 1.0

    if len(img_new.shape) == 3:
        img_new = img_new[None]  # expand for batch dim
    pred = model(img_new, augment=False, visualize=False)
    pred = non_max_suppression(pred, max_det=1000)[0]

    s = '%gx%g ' % img_new.shape[2:]  # print string

    if len(pred):
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_boxes(img_new.shape[2:], pred[:, :4], img_ori.shape).round()  # todo:0:
        # pred: 左上xy,右下xy,confidence,class
        shape = img_ori.shape
        img_ori = Image.fromarray(img_ori[:, :, ::-1])
        for *xyxy, confidence, cls in reversed(pred):  # todo:坐标置信度。类别
            font = ImageFont.truetype(font='simsun.ttc',
                                      size=np.floor(1e-3 * shape[1] + 15).astype('int32'), index=1)
            draw = ImageDraw.Draw(img_ori)
            draw.rectangle(xyxy, outline='red', width=2)
            plate_img = np.array(img_ori.crop(np.array(xyxy)))[:, :, ::-1]
            plate_text = demo.recognition(text_path, plate_img)
            text_border(draw, xyxy, ("置信度%.2f%%\n" % (confidence * 100).data) + plate_text, font,
                        shadowcolor=(0, 0, 0), fillcolor='white')


if __name__ == '__main__':
    main("weights/yolo.pt", "weights/rec_best.pth", 'zq.jpg')
