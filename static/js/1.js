$(".img-content").click(
    (item) => {
        stopVideo(video);
        contentImg = new Image();
        contentImg.src = item.currentTarget.src;

        contentImg.onload = function() {
            faceAnalysis(contentImg, canvasTemp);
        }
    })


$('#openCameraBt').click(() => {
    startVideo(video).then( x=> faceVideoAnalysis(video));
});


$('#switchCameraBt').click(() => {
    startVideo(video,switchCamera=true).then(()=> {
        faceVideoAnalysis(video);
    });
})

$('#closeCameraBt').click(()=>stopVideo(video));

$('#saveImgBt').click(() => {
    let downloadLink = document.createElement("a");
    downloadLink.download = 'face.png';
    downloadLink.href = canvasDownload.toDataURL("image/png");
    downloadLink.click();
    downloadLink.remove();
})

$(".switchBt").click( (item) => {
        if (item.target.id == "showBox") {
            showBox = item.target.checked;
        } else if(item.target.id == "showLandmark") {
            showLandmark = item.target.checked;
        }
        else if (item.target.id == "showExpression") {
            showExpression = item.target.checked;
        }
        else if (item.target.id == "showAgeGender") {
            showAgeGender = item.target.checked;
        }
        faceAnalysis(contentImg);
    }
)



$('#contentImgUrlConfirmBt').click(() =>{
    let value = $('#contentImgUrlInput').val()
    // contentImgTemp = new Image();
    contentImg.crossOrigin = "anonymous";
    contentImg.src = value;
    contentImg.onload = async () => {
        //   console.log("图像加载成功");
        faceAnalysis(contentImg);

    }
    contentImg.onerror = () => {alert("您输入的URL被拒绝访问，请换一张图片～");}
})


$('#contentImgFileUploadInput').on('change', ()=> {
    let fileObj = $('#contentImgFileUploadInput').prop('files')[0];
    contentImg.onload = async () => {
        console.log("上传内容图像加载成功");
        faceAnalysis(contentImg);
        // segmentAndCombine(contentImg, contentImg);
    }
    loadImage(
        fileObj,
        (img) => { contentImg.src = img.toDataURL();
            contentImg.width = img.width;
            contentImg.height = img.height;
            console.log("Img width:", contentImg.width, contentImg.height);
        },
        { orientation:true,
            maxWidth:480,
        }
    );
})




let contentImg;
let showBox = true;
let showLandmark = true;
let showExpression = true;
let showAgeGender = true;
let intervalID;
let canvasDownload = document.createElement('canvas'); //作为下载图片专用的canvas
let canvasDownloadContext = canvasDownload.getContext('2d');

let resultImg = new Image();
let canvas_show = document.getElementById("canvas_show");
let context = canvas_show.getContext('2d');
let canvasTemp = document.createElement("canvas");

const modelRepoAddr = "https://file.aizoo.com/model/cv/facejs";
contentImgDefault = document.getElementsByClassName("img-content")[5];

contentImg = new Image();
contentImg.src = contentImgDefault.src;

en2zh = {'angry':'生气', 'disgusted':'讨厌', 'fearful':'害怕' ,'happy':'开心','neutral':'中性','sad':'忧伤','surprised':'惊喜'}

async function loadModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelRepoAddr);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(modelRepoAddr);
    await faceapi.nets.faceLandmark68Net.loadFromUri(modelRepoAddr);
    await faceapi.nets.ageGenderNet.loadFromUri(modelRepoAddr);
    await faceapi.nets.faceExpressionNet.loadFromUri(modelRepoAddr);
}

// 找出最大的表情置信度
function findMaxExpression(expression) {
    let maxExpression;
    let maxScore = -1;
    Object.keys(expression).map(x => {
        if (expression[x] > maxScore) {
            maxExpression = x;
            maxScore = expression[x];
        }
    })
    return [en2zh[maxExpression], maxScore];
}

// 返回将结果画到图上的img
async function faceAnalysisInternal(img, canvas, drawBox=true, landmark=true, expression=true, ageGender=true) {
    resultPromise = faceapi.detectAllFaces(img);
    if (landmark) {
        resultPromise = resultPromise.withFaceLandmarks();
    }
    if (expression) {
        resultPromise = resultPromise.withFaceExpressions();
    }
    if (ageGender) {
        resultPromise = resultPromise.withAgeAndGender();
    }
    results = await resultPromise;
    canvas.width = img.width;
    canvas.height = img.height;
    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    if (drawBox) {
        faceapi.draw.drawDetections(canvas, results);
    }
    if (landmark) {
        faceapi.draw.drawFaceLandmarks(canvas, results);
    }

    if (ageGender || expression) {
        results.forEach(result => {
            text = []
            if (ageGender) {
                let age = result['age'];
                let gender = result['gender'] === 'female' ? '女孩':'男';
                let genderProbability = result['genderProbability'];
                text.push( `${faceapi.utils.round(age, 0)} 岁`, `${gender} (${faceapi.utils.round(genderProbability)})`);
            }
            if (expression) {
                let expr = result['expressions'];
                let exprMax = findMaxExpression(expr);
                let expressionZh = exprMax[0];
                let expressionScore = faceapi.utils.round(exprMax[1]);
                text.push(expressionZh + "(" + expressionScore + ")")
            }
            new faceapi.draw.DrawTextField(
                text,
                result.detection.box.bottomLeft
            ).draw(canvas)
        })
    }
    return canvas;
}



function writeToCanvasDownload(img) {
    canvasDownload.width = img.width;
    canvasDownload.height = img.height;
    canvasDownloadContext.clearRect(0, 0, canvasDownload.width, canvasDownload.height);
    canvasDownloadContext.drawImage(img, 0, 0, img.width, img.height)

}

async function faceAnalysis(img) {
    await faceAnalysisInternal(img, canvasTemp, showBox, showLandmark, showExpression, showAgeGender);
    resultImg.src = canvasTemp.toDataURL();
    resultImg.onload = () => {
        writeToCanvasDownload(resultImg);
        cords = calculateLocationInCanvas(canvas_show.width,
            canvas_show.height, resultImg.width, resultImg.height);
        // console.log(cords);
        context.clearRect(0, 0, canvas_show.width, canvas_show.height);
        context.drawImage(resultImg , cords[0], cords[1], cords[2], cords[3]);
        // })
    }
}

function faceVideoAnalysis(video) {
    if (!video.paused) {
        faceAnalysis(video);
        intervalID = setTimeout(() => {
            faceVideoAnalysis(video);
        });
        // faceVideoAnalysis();
    }
}

async function setup() {
    await loadModels();
    video = document.createElement('video');
    video.width = 400;
    video.height = 400;
    video.setAttribute("style", "display: none;");
    document.body.appendChild(video);
    await faceAnalysis(contentImg);
    $('#loadModelIndictor').hide();
}

//   getAllCamera(videoSources); // 获取所有的摄像头ID
setup();