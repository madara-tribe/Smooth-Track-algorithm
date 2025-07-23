import argparse
import cv2
import time
import sys
import numpy as np
import random
from onnx_utils import YOLODetect

TARGET_CANDIDATE_CLS = ['clock'] #, 'cell phone']#  'tv', 'remote', 'cup']
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--yolo_onnx_path', '-yp', type=str, default='weights/yolov7Tiny_640_640.onnx')
    return parser.parse_args()

class SmoothTrack:
    def __init__(self, opt):
        self.opt = opt
        self.prev_centers = {}
        self.single_mode = True
        
    def clip_box(self, x1, y1, x2, y2, img_w, img_h):
        return (
            max(0, min(x1, img_w - 1)),
            max(0, min(y1, img_h - 1)),
            max(0, min(x2, img_w - 1)),
            max(0, min(y2, img_h - 1))
        )
    def bbox_diff(self, box, label):
        center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        diff_x, diff_y = (center_x - self.prev_centers[label][0], center_y - self.prev_centers[label][1]) if label in self.prev_centers else (0, 0)
        self.prev_centers[label] = (center_x, center_y)
        moved = [box[0] + diff_x, box[1] + diff_y, box[2] + diff_x, box[3] + diff_y]
        return moved
        
    def track(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        detector = YOLODetect(self.opt)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                outputs, ori_images, ratio, dwdh, conf_thres = detector.inference_(frame)
                image = ori_images[0].copy()
                boxes = []

                if self.single_mode and len(outputs) > 0:
                    outputs = [sorted(outputs, key=lambda x: x[6], reverse=True)[0]]

                for det in outputs:
                    x1, y1, x2, y2, cls_id, score = det[1:7]
                    score = float(score)
                    if score < conf_thres:
                        continue

                    name = names[int(cls_id)]
                    if name not in TARGET_CANDIDATE_CLS:
                        continue

                    box = np.array([x1, y1, x2, y2]) - np.array(dwdh * 2)
                    box /= ratio
                    box = box.round().astype(int).tolist()
                    label = f"{name} {round(score, 2)}"
                    
                    moved = self.bbox_diff(box, label)
                    img_h, img_w = image.shape[:2]
                    moved = self.clip_box(*moved, img_w, img_h)

                    color = colors.get(name, (0, 255, 0))
                    cv2.rectangle(image, (moved[0], moved[1]), (moved[2], moved[3]), color, 2)
                    cv2.putText(image, label, (moved[0], max(0, moved[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    boxes.append(moved)

                if not self.single_mode and len(boxes) > 1:
                    x1s, y1s, x2s, y2s = zip(*boxes)
                    group_box = self.clip_box(min(x1s), min(y1s), max(x2s), max(y2s), img_w, img_h)
                    cv2.rectangle(image, (group_box[0], group_box[1]), (group_box[2], group_box[3]), (255, 255, 0), 2)
                    cv2.putText(image, "Group", (group_box[0], max(0, group_box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                print("Prediction took {:.2f} sec".format(time.time() - start_time))
                cv2.imshow("Camera", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("Shutdown complete.")


if __name__ == "__main__":
    opt = get_parser()
    smooth_tracking = SmoothTrack(opt)
    smooth_tracking.track()

