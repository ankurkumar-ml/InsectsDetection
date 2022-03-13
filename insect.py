import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.common import DetectMultiBackend
from yolo.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolo.utils.plots import Annotator, colors, save_one_box
from yolo.utils.torch_utils import select_device, time_sync
import config

import logging
logging.getLogger("yolo.utils.general").setLevel(logging.ERROR)


@torch.no_grad()
def run(weights=config.insect_weight_path,  # model.pt path(s)
        # For stamp detect replace sign_test2.pt to stamp_detect.pt in above path
        # sign_detect, sign_test1, sign_test2 All are for sign detection improved models in sequence.
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(1280, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / r'C:\Users\91911\Documents\rezo.ai\Fusion_microservice\row+stamp_check_data\checked',
        # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,
        dataframe=None,  # use OpenCV DNN for ONNX inference
        src_img_path=None,
        save_to_img_path=None
        ):
    data_dir = src_img_path
    filename = source
    source = os.path.join(data_dir, source)

    raw_img = cv2.imread(source)

    # print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = save_to_img_path
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            insect_count = 0
            egg_count = 0
            conf_thresh = 0.2


            p = Path(p)  # to Path
            save_path = os.path.join(save_dir, filename)  # im.jpg
            txt_path = str(save_dir) + "/" + 'labels' "/" + str(p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        label = label.split(" ")[0]
                        #print("Label: {}  Conf: {}".format(label, conf))
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        ##For row count in the cgt sheets.

                        if label == 'I' and conf.item() > conf_thresh:
                            insect_count += 1

                        if label == 'E' and conf.item() > conf_thresh:
                            egg_count += 1


                dataframe.loc[len(dataframe.index)] = [filename, insect_count, egg_count]

            # Print time (inference-only)
            #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            # write counts
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            # fontScale
            fontScale = 2
            # Blue color in BGR
            color = (0, 255, 0)
            thickness = 3
            # Using cv2.putText() method
            im0 = cv2.putText(im0, 'Total Insects: {}    Total Eggs: {}'.format(insect_count, egg_count), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def show_img(img):
    cv2.imshow("hi", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def insect_prediction(filename, src_img_path, dst_img_path):
    df = pd.DataFrame(columns=["filename", "insect_count", "egg_count"])
    run(source=filename, dataframe=df, src_img_path=src_img_path, save_to_img_path=dst_img_path)

    insect_count = list(df['insect_count'])[0]
    egg_count = list(df['egg_count'])[0]
    counts = [insect_count, egg_count]
    return filename, counts

def multiple_files():
    data_path = "C:\\Users\\ankur\\Downloads\\NewImages\\tmp\\images"
    files = os.listdir(data_path)
    #print(files)
    df = pd.DataFrame(columns=["Filename", "insect_count", "egg_count"])
    #print(files)
    count = 0
    for file in files:
        run(source=file, dataframe=df)
        #print("{} --> done".format(file))
        count += 1
        print(f"Done: {count}", end="\r")

    print("Done")
    import datetime as dt
    df['DateTime'] = dt.datetime.now()
    df.to_excel(r"C:\\Users\\ankur\\Downloads\\NewImages\\tmp\\bees_count_result.xlsx")
