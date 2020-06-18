// 计算图像在canvas中显示的起点和长宽
function calculateLocationInCanvas(canvasWidth, canvasHeight, imgWidth, imgHeight) {
    let x, y, newHeight, newWidth;
    if (imgWidth <= canvasWidth) {
        if (imgHeight <= canvasHeight) {
            // 这种方式仅显示了一个小图像，没有把整个canvas占满
            // x = (canvasWidth - imgWidth) / 2;
            // y = (canvasHeight - imgHeight) / 2;
            if (imgHeight / imgWidth <= canvasHeight / canvasWidth) {
                // 图像是更宽的
                x = 0;
                newWidth = canvasWidth;
                newHeight = imgHeight * canvasWidth / imgWidth;
                y = (canvasHeight - newHeight) / 2 ;
            } else {
                // 图像是更高的
                y = 0;
                newHeight = canvasHeight;
                newWidth = imgWidth * canvasHeight / imgHeight;
                x = (canvasWidth - newWidth)/2;
            }
            return [x, y, newWidth, newHeight]
        } else {
            newWidth = imgWidth * canvasHeight / imgHeight;
            x = (canvasWidth - newWidth) / 2;
            return [x, 0, newWidth, canvasHeight]
        }

    } else {
        if (imgHeight <= canvasHeight) {
            newHeight = imgHeight * canvasWidth / imgWidth;
            y = (canvasHeight - newHeight) / 2;
            return [0, y, canvasWidth, newHeight]
        } else { // 如果img长和宽都大于canvas的宽度，这种情况比较复杂
            if (imgHeight/imgWidth <= canvasHeight/canvasWidth) {
                newHeight = imgHeight * canvasWidth / imgWidth;
                y = (canvasHeight - newHeight) / 2;
                return [0, y, canvasWidth, newHeight];
            } else {
                newWidth = imgWidth * canvasHeight / imgHeight;
                x = (canvasWidth - newWidth) / 2;
                return [x, 0, newWidth, canvasHeight]
            }
        }
    }
}

// 获取所有的摄像头设备
async function getAllCamera() {
    deviceIDs = [];
    allDevices = await navigator.mediaDevices.enumerateDevices();
    for (let device of allDevices) {
        if (device.kind == 'videoinput') {
            deviceIDs.push(device.deviceId);
        }
    }
    return deviceIDs;
}


async function startVideo(videoElement, switchCamera=false, videoSizeConfig={}) {
    // 下面关于长宽必须指定的要求存疑，好像不指定也可以，打印出来长宽都是0
    // 如果有已经打开的摄像头，先关闭它
    videoSources = await getAllCamera();
    if (switchCamera) { // 需要切换前后摄像头
        console.log("switch camera is true")
        if (videoSources.length == 1) { // 只有一个摄像头，无法切换
            alert("您的设备上只有一个摄像头，无需切换前后摄像头");
            return;
        } else  { // 有两个以上的摄像头，可以切换
            if (videoElement.srcObject) { // 如果有打开的，则关闭当前的设备
                videoElement.srcObject.getTracks().forEach(track => {
                    track.stop();
                });
            }
            if (window.cameraID === undefined) {
                window.cameraID = 0;
            } else {
                window.cameraID = window.cameraID + 1;
            }
            videoSource = videoSources[window.cameraID % videoSources.length];
        }
    } else { // 是第一次打开摄像头
        if (videoSources.length == 0) { // 只有一个摄像头，无法切换
            alert("您的设备上没有摄像头，或者您没有赋予浏览器获取摄像头的权限");
            return;
        } else  {
            window.cameraID = 0; // 给窗体增加一个全局的变量，标记camera的ID
            videoSource = videoSources[0];
        }
    }
    videoConfig = {
        audio: false,
        video: {deviceId: videoSource ? {exact: videoSource} : undefined,
            width:416, height:416}
    }
    return new Promise(function (resolve, reject) {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(videoConfig)
                .then(stream => {
                    console.log("本次ID：", videoSource)
                    window.stream = stream;
                    videoElement.srcObject = stream
                    videoElement.onloadeddata = () => {
                        videoElement.play()
                        resolve(true)
                    }
                }).catch(function (err) {
                resolve(false)
            });
        }});
}


async function stopVideo(videoElement) {
    if (videoElement.srcObject) {
        videoElement.pause();
        videoElement.srcObject.getTracks().forEach((track) => {
            track.stop();
        });
        videoElement.srcObject = null;
        return true;
    } else {
        return false;
    }
}





