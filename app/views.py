import json
import os

from django.http import HttpResponse
from django.shortcuts import render
from detector import detect

det = detect.Detect()


def hello(request):
    return render(request, 'index.html')


def upload(request):
    data = {}
    if request.method == "POST":
        fp = request.FILES.get("file")
        # fp 获取到的上传文件对象
        if fp:
            path = os.path.join('static/', 'img/' + fp.name)  # 上传文件本地保存路径， image是static文件夹下专门存放图片的文件夹
            # fp.name #文件名
            # yield = fp.chunks() # 流式获取文件内容
            # fp.read() # 直接读取文件内容
            if fp.multiple_chunks():  # 判断上传文件大于2.5MB的大文件
                # 为真
                file_yield = fp.chunks()  # 迭代写入文件
                with open(path, 'wb') as f:
                    for buf in file_yield:  # for情况执行无误才执行 else
                        f.write(buf)
                    else:
                        data['code'] = 1
            else:
                with open(path, 'wb') as f:
                    f.write(fp.read())
                print("小文件上传完毕")
            # path = os.path.abspath(path)

            data['code'] = 1
            data['path'] = det.detect(path)
        else:
            data['code'] = 0
    return HttpResponse(json.dumps(data), content_type="application/json,charset=utf-8")
