
from detector.models import *
from detector.utils.datasets import *
from detector.utils.utils import *


class Detect:
    def __init__(self, ):
        self.cfg = 'detector/cfg/yolov3_person-anchors.cfg'
        self.half = True
        self.out = 'static/img'
        self.img_size = 416
        self.weights = 'detector/weights/best7.pt'
        self.names = 'detector/data/person/person.names'
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.agnostic_nms = False
        self.fourcc = 'mp4v'
        self.version = 1
        self.load_model()

    def load_model(self):
        print("加载权重文件")
        self.model = Darknet(self.cfg, self.img_size)
        self.device = torch_utils.select_device(device='cuda' if torch.cuda.is_available() else 'cpu')

        if self.weights.endswith('.pt'):  # pytorch 格式
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet 格式
            load_darknet_weights(self.model, self.weights)
        self.model.to(self.device).eval()
        self.half = self.half and self.device.type != 'cpu'  # 半精度只支持cuda
        if self.half:
            self.model.half()
        # 获取names文件和随机颜色框
        self.names = load_classes(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def change_cfg(self, img_size, conf_thres, iou_thres, version):
        load = False
        if version != self.version:
            self.version = version
            if version == 1:
                self.weights = 'detector/weights/best7.pt'
                self.cfg = 'detector/cfg/yolov3_person-anchors.cfg'
                self.names = 'detector/data/person/person.names'
            elif version == 2:
                self.weights = 'detector/weights/best8.pt'
                self.cfg = 'detector/cfg/yolov3_person-se.cfg'
                self.names = 'detector/data/person/person.names'
            elif version == 3:
                self.weights = 'detector/weights/best9.pt'
                self.cfg = 'detector/cfg/yolov3_person-spp.cfg'
                self.names = 'detector/data/person/person.names'
            elif version == 4:
                self.weights = 'detector/weights/yolov3.weights'
                self.cfg = 'detector/cfg/yolov3.cfg'
                self.names = 'detector/data/coco.names'
            load = True
        if img_size != self.img_size and img_size % 32 == 0 and 0 < img_size < 609:
            self.img_size = img_size
            load = True
        if 0 < conf_thres < 1:
            self.conf_thres = conf_thres
        if 0 < iou_thres < 1:
            self.iou_thres = iou_thres
        if load:
            self.load_model()

    def detect_binary(self, img, ws, is_seq=True):
        with torch.no_grad():
            # dataset = LoadImages(img, img_size=self.img_size)
            result = []
            # 开始推理

            img0 = img if is_seq else cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)  # BGR
            assert img0 is not None, 'Image Not Found '
            img = letterbox(img0, new_shape=self.img_size)[0]
            # 转码
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            # for path, img, im0s, vid_cap in dataset:
            t = time.time()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 推理
            pred = self.model(img)[0].float() if self.half else self.model(img)[0]

            # 使用 NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       agnostic=self.agnostic_nms)

            # 处理检测结果
            for i, det in enumerate(pred):
                s, im0 = '', img0

                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, time.time() - t))
                result.append({})
                result[i]["time"] = time.time() - t
                result[i]["path"] = im0

            ws.send(bytes_data=cv2.imencode('.jpg', im0)[1].tobytes())

    # def detect_seq(self, ws):
    #     cap = cv2.VideoCapture('static/temp.seq')
    #     ret_val, img0 = cap.read()
    #     while ret_val:
    #         self.detect_binary(img0, ws)
    #         ret_val, img0 = cap.read()
    #     cap.release()

    def detect_file(self, ws, file_path='static/temp.seq'):

        if os.path.exists(file_path):
            if os.path.splitext(file_path)[-1].lower() in vid_formats:
                cap = cv2.VideoCapture(file_path)
                ret_val, img0 = cap.read()
                while ret_val and not ws.stop:
                    self.detect_binary(img0, ws)
                    ret_val, img0 = cap.read()
                cap.release()
