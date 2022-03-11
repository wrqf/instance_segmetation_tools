#https://download.csdn.net/download/qq_34510308/12087286
import sys
import os
import cv2
import numpy as np
import random
import time
import json
import base64

from math import cos,sin,pi,fabs,radians

# 读取添加模板
path = './image/'
"""
classes = ["Half open window", "Lighting doors windows", "Open door window_singular_right","Open door window_singular_left",
           "Multiple open doors windows_no direction","Multiple open doors windows_left","Multiple open doors windows_right",
           "Horizontal sliding door window_singular_right",
           "Horizontal sliding door window_singular_left",
           "Horizontal sliding door window_no direction","Folding doors windows_singular_right",
           "Folding doors windows_singular_left","Folding doors windows_no direction"]

label1 = ['Multiple open doors windows_no direction', 'Multiple open doors windows_left',
          'Multiple open doors windows_right', 'Folding doors windows_singular_right',
          'Folding doors windows_singular_left', 'Folding doors windows_no direction']
"""
multiLabel_r=['Open door window_singular_right','Open door window_singular_right_1','Open door window_singular_right_s90']
multiLabel_l=['Open door window_singular_left','Open door window_singular_left_1','Open door window_singular_left_2']



label2 = ['Lighting doors windows', multiLabel_r, multiLabel_l,
          'Half open window', 'Horizontal sliding door window_singular_right',
          'Horizontal sliding door window_singular_left',"Horizontal sliding door window_no direction",
          'Multiple open doors windows_no direction', 'Multiple open doors windows_left',
          'Multiple open doors windows_right', 'Folding doors windows_singular_right',
          'Folding doors windows_singular_left', 'Folding doors windows_no direction'
          ]
# label2=['Open door window_singular_left_2']
bc_random_xy = [100, 130, 200, 250]  # 标尺左上点的随机位置范围，50～70标尺1左上点的x的范围，200~250是标尺1左上点的y的范围。标尺2类似。
ht_random_xy = [10, 40]  # 贴图离边缘的随机范围
w_random = [1600, 3000]  # 新建图的宽
h_random = [1400, 2800]  # 新建图的高
bili_list = [0.8, 1.3]  # bili > 0.8 and bili < 1.3
size_random=[0.8, 2.0] #随机resize大小的比例系数
bc_list=['./biaochi1.jpg','./biaochi2.jpg']#标尺图片
biaochi_bili=[1.2,1.6]  #标尺比例

#读取json
def readJson(jsonfile):
    with open(jsonfile,encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData

#保存json
def writeToJson(filePath,data):
    fb = open(filePath,'w')
    fb.write(json.dumps(data,indent=2)) # ,encoding='utf-8'
    fb.close()

def hetu_biaochi(image_bg, imgge):
    """
    画标尺
    :param image_bg:白色背景图
    :param imgge:标尺模板图
    :return:
        randx：标尺在背景图上的左上点x坐标
        cols：标尺的宽度
        randy：标尺在背景图上的左上点y坐标
        rows：标尺的高度
    """
    moban_rows, moban_cols, moban_channels = imgge.shape
    ImageBg_rows,ImageBg_cols,ImageBg_channels=image_bg.shape
    if (moban_rows > moban_cols):  # 是标尺1
        randx = random.randint(bc_random_xy[0], bc_random_xy[1])
        randy = random.randint(bc_random_xy[2], bc_random_xy[3])
        # print("111",randx,randy)
        # print(moban_rows/ImageBg_rows)
        if moban_rows/ImageBg_rows<0.5:
            size=random.uniform(biaochi_bili[0],biaochi_bili[1])
            imgge = cv2.resize(imgge, None, fx=1.0, fy=size)
        else:
            pass
    elif (moban_rows < moban_cols):  # 是标尺2
        # print("222", randx, randy)
        if moban_cols/ImageBg_cols<0.5:
            size=random.uniform(biaochi_bili[0],biaochi_bili[1])
            imgge = cv2.resize(imgge, None, fx=size, fy=1.0)
        else:
            pass
        moban_rows, moban_cols, moban_channels = imgge.shape
        randx = random.randint(image_bg.shape[1] - moban_cols - bc_random_xy[3], image_bg.shape[1] - moban_cols - bc_random_xy[2])
        randy = random.randint(image_bg.shape[0] - moban_rows - bc_random_xy[1], image_bg.shape[0] - moban_rows - bc_random_xy[0])

    rows, cols, channels = imgge.shape
    # print(randx,randy,rows + randy,cols + randx,imgge.shape,image_bg.shape)
    roi = image_bg[randy:rows + randy, randx:cols + randx]
    # print("cols:",cols,"rows:",rows,"randy:",randy,"randx:",randx)
    # print(roi.shape,imgge.shape,image_bg.shape)
    # cv2.imshow("roi",roi)
    # cv2.imshow("imgge",imgge)
    # cv2.waitKey(0)
    imggray = cv2.cvtColor(imgge, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 150, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img_fg = cv2.bitwise_and(imgge, imgge, mask=mask_inv)
    dst = cv2.add(img_bg, img_fg)
    image_bg[randy:rows + randy, randx:cols + randx] = dst
    return randx, cols, randy, rows

def hetu(bc_x, bc_y, image_bg, imgge):
    """
    :param bc_x: 左边离标尺的距离
    :param bc_y: 下边离标尺的距离
    :param image_bg: 背景白图
    :param imgge: 需要贴的图
    :return:
        randx：贴图在背景图上的左上点x坐标
        cols：贴图的宽度
        randy：贴图在背景图上的左上点y坐标
        rows：贴图的高度
    """
    rows, cols, channels = imgge.shape
    randx = random.randint(bc_x + ht_random_xy[0], image_bg.shape[1] - cols - ht_random_xy[1])
    randy = random.randint(ht_random_xy[1], image_bg.shape[0] - bc_y - rows - ht_random_xy[0])
    roi = image_bg[randy:rows + randy, randx:cols + randx]

    imggray = cv2.cvtColor(imgge, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 150, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img_fg = cv2.bitwise_and(imgge, imgge, mask=mask_inv)
    dst = cv2.add(img_bg, img_fg)
    image_bg[randy:rows + randy, randx:cols + randx] = dst
    return randx, cols, randy, rows


def compute_iou(rec1, rec2):
    '''
    计算IOU
    :param rec1:
    :param rec2:
    :return:
        iou：
    '''
    cy1, cx1, cy2, cx2 = rec1
    gy1, gx1, gy2, gx2 = rec2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def genrateImage():
    # 生成两张图片
    bili = 1.0
    while (True):
        # 创建纯白图像
        _w = random.randint(w_random[0], w_random[1])
        _h = random.randint(h_random[0], h_random[1])
        bili = _w / _h
        if (bili > bili_list[0] and bili < bili_list[1]):
            break
        else:
            continue
    return _w, _h

def morphologyImage(temp):
    """
    形态学处理
    :param temp:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 矩形结构
    erodeImage = cv2.erode(temp, kernel)  # 膨胀
    return erodeImage

def GaussProcess(temp):
    """
    高斯模糊
    :param temp:
    :return:
    """
    gaussImage = cv2.GaussianBlur(temp, (3, 3), 0)
    return gaussImage


def rotate_bound(image, angle):
    """
    旋转图像
    :param image: 图像
    :param angle: 角度
    :return: 旋转后的图像
    """
    h, w,_ = image.shape
    # print(image.shape)
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image_rotate = cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    return image_rotate,cX,cY,angle


def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,matRotation


#旋转后的坐标变换
def rotate_xy(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    x2 = (x1-x0)*cosA - (y1-y0)*sinA + x0
    y2 = (x1-x0)*sinA + (y1-y0)*cosA + y0
    """
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new



#坐标转换
def TransforPoint(jsonTemp, scale):
    json_dict = {}
    for key, value in jsonTemp.items():
        json_dict[key]=value

    for item in  json_dict['shapes']:
        for item2 in item['points']:
            item2[0],item2[1]=item2[0]*scale,item2[1]*scale
            # print(item2[0],item2[1])
    return json_dict



#坐标旋转
def rotatePoint(Srcimg_rotate,jsonTemp,M):
    json_dict = {}
    for key, value in jsonTemp.items():
        if key=='imageHeight':
            json_dict[key]=Srcimg_rotate.shape[0]
        elif key=='imageWidth':
            json_dict[key] = Srcimg_rotate.shape[1]
        elif key=='imageData':
            json_dict[key] = image_to_base64(Srcimg_rotate)
        elif key=='imagePath':
            json_dict[key] = 'res1_%d.jpg' % 1
        else:
            json_dict[key] = value
    for item in json_dict['shapes']:
        for key, value in item.items():
            if key == 'points':
                for item2 in range(len(value)):
                    pt1=np.dot(M,np.array([[value[item2][0]],[value[item2][1]],[1]]))
                    value[item2][0], value[item2][1] = pt1[0][0], pt1[1][0]
    return json_dict

#图像在大图上的坐标
def TransforPoint2imag(jsonTemp,box):
    for jsonList, boxList in zip(jsonTemp, box):
        Imag_x=boxList[0][1]
        Imag_y=boxList[0][0]
        # print(boxList,Imag_x,Imag_y)
        for item in jsonList['shapes']:
            for key, value in item.items():
                if key=='points':
                    for item2 in range(len(value)):
                        value[item2][0],value[item2][1]=value[item2][0]+Imag_x,value[item2][1]+Imag_y



def read_img(i,ImgArray,josnList,h=0):
    """
    读取图片
    :param i:第i种图
    :param ImgArray:存放读取图像
    :return:
    """
    # img_box = []
    # json_enum=[]
    if  i==1 or i==2:
        print(label2[i][h])
        Srcimg = cv2.imread(path + str(label2[i][h]) + '.jpg')  ##########gai label1,label2
        jsonData = readJson(path + str(label2[i][h]) + '.json')  ######## 读取json
    else:
        Srcimg = cv2.imread(path + str(label2[i]) + '.jpg')  ##########gai label1,label2
        jsonData = readJson(path + str(label2[i]) + '.json')   ######## 读取json
    m=random.randint(0,2)
    if m==1:
        img=Srcimg
    elif m==2:
        img=morphologyImage(Srcimg)
    else:
        img=GaussProcess(Srcimg)
    n = random.randint(0, 2)
    if n==0:
        img_end = img
        json_end = jsonData
    elif n==1:
        img_s90, M = dumpRotateImage(Srcimg, 90)
        jsonData2 = rotatePoint(img_s90, jsonData, M)
        img_end=img_s90
        json_end=jsonData2
    else:
        img_n90, M2 = dumpRotateImage(Srcimg, 270)
        jsonData3 = rotatePoint(img_n90, jsonData, M2)
        img_end = img_n90
        json_end = jsonData3


    size = random.uniform(size_random[0],size_random[1])
    jsonArray=TransforPoint(json_end,size)
    josnList.append(jsonArray)
    img_end = cv2.resize(img_end, None, fx=size, fy=size)
    ImgArray.append(img_end)


#转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


#generator json file
def generatorJson(num,width,height,jsonList,img):
    json_dict = {}
    label_dict = {'shapes': []}
    for item in jsonList:
        for item2 in item['shapes']:
            temp_dict = {}
            for key, value in item2.items():
                temp_dict[key] = value
            label_dict['shapes'].append(temp_dict)
    for key, value in jsonList[0].items():
        if key == 'shapes':
            json_dict[key] = label_dict['shapes']
        elif key=='imageHeight':
            json_dict[key]=height
        elif key=='imageWidth':
            json_dict[key] = width
        elif key=='imagePath':
            json_dict[key]='res1_%d.jpg'%num
        elif key=='imageData':
            json_dict[key]=image_to_base64(img)
        else:
            json_dict[key]=value
    writeToJson('./result3/res1_%d.json'%num, json_dict)






def kuo(num, img_w, img_h):
    """
    扩充数据
    :param num: 第num张图
    :param img_w:整个图的宽
    :param img_h:整个图的高
    :return:
    """
    ImgArray = []
    labelname = []
    jsonList=[]

    ###############################gai#######
    for i in range(6):
        mclasses = random.randint(0, 12)
        # print(mclasses)
        if mclasses == 1 or mclasses == 2:
            lable_num = random.randint(1, 4)
            for ii in range(lable_num):
                h = random.randint(0, 2)
                read_img(mclasses, ImgArray,jsonList,h)
                labelname.append(label2[mclasses][0])
        else:
            lable_num = random.randint(1, 3)  # 一张图上画的张数 随机
            for ii in range(lable_num):
                read_img(mclasses, ImgArray,jsonList)
                labelname.append(label2[mclasses])



    ###############################gai##########

    # 读取标尺
    img_biaochi_1 = cv2.imread(bc_list[0])
    img_biaochi_2 = cv2.imread(bc_list[1])

    flag = True
    while (flag):
        img = np.zeros([img_h, img_w, 3], np.uint8)
        img[:, :, 0] = np.zeros([img_h, img_w]) + 255
        img[:, :, 1] = np.ones([img_h, img_w]) + 254
        img[:, :, 2] = np.ones([img_h, img_w]) * 255

        # 画标尺
        bc_x1, bc_cols1, bc_y1, bc_rows1 = hetu_biaochi(img, img_biaochi_1)
        bc_x2, bc_cols2, bc_y2, bc_rows2 = hetu_biaochi(img, img_biaochi_2)

        tempRect = []
        box = []
        bc_x = bc_x1 + bc_cols1
        bc_y = img.shape[0] - bc_y2
        for i in range(len(ImgArray)):
            randx1, cols1, randy1, rows1 = hetu(bc_x, bc_y, img, ImgArray[i])
            box.append([(randy1, randx1), (randy1 + rows1, randx1 + cols1)])
            tempRect.append((randy1, randx1, randy1 + rows1, randx1 + cols1))
        for i in range(len(tempRect) - 1):
            flag_iou = True
            for j in range(i + 1, len(tempRect)):
                iou = compute_iou(tempRect[i], tempRect[j])
                if iou > 0.1:
                    flag_iou = False
                    break
                else:
                    continue
            if (flag_iou == False):
                flag = True
                break
            else:
                flag = False
    TransforPoint2imag(jsonList,box)  ##转换坐标,将模板坐标转成图像坐标
    height, width, _ = img.shape
    cv2.imwrite('./result3/res1_%d.jpg' % num, img)  ######################gai res1,res2
    ##json
    generatorJson(num,width,height,jsonList,img)


if __name__ == '__main__':
    for num in range(10):
        img_w, img_h = genrateImage()
        kuo(num, img_w, img_h)
        print('%d OK!' % num)
    print('END!')
