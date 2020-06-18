import json
import threading

from channels.generic.websocket import WebsocketConsumer
from app.views import det


class Consumer(WebsocketConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = True

    def connect(self):
        print('连接成功')
        self.accept()

    def disconnect(self, close_code):
        self.stop = True
        print('失去连接')

    def receive(self, text_data=None, bytes_data=None):
        if text_data:
            js = json.loads(text_data, encoding='utf-8')
            print(js)
            if 'img_size' in js:
                det.change_cfg(js['img_size'], js['conf_thres'], js['iou_thres'], js['weights'])
            if 'local_file' in js :
                if js['local_file'] != 1 and js['local_file'] != 2:
                    self.stop = False
                    threading.Thread(target=det.detect_file, args=(self, js['local_file'])).start()
            if 'stop' in js :
                self.stop=True
            self.send(text_data)
        elif bytes_data:
            if len(bytes_data) > 1024 * 1024 * 8:
                with open('static/temp.seq', 'wb') as f:
                    f.write(bytes_data)
                det.detect_file(self)
            else:
                det.detect_binary(bytes_data, self, False)
