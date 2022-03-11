#https://download.csdn.net/download/qq_34510308/12087286
import sys
import os
import cv2
import numpy as np
import random
import time

# 读取添加模板
path = './image/'
classes = ["Half open window", "Lighting doors windows", "Open door window_singular_right","Open door window_singular_left",
           "Multiple open doors windows_no direction","Multiple open doors windows_left","Multiple open doors windows_right",
           "Horizontal sliding door window_singular_right",
           "Horizontal sliding door window_singular_left",
           "Horizontal sliding door window_no direction","Folding doors windows_singular_right",
           "Folding doors windows_singular_left","Folding doors windows_no direction"]
"""
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

bc_random_xy = [100, 130, 200, 250]  # 标尺左上点的随机位置范围，50～70标尺1左上点的x的范围，200~250是标尺1左上点的y的范围。标尺2类似。
ht_random_xy = [10, 40]  # 贴图离边缘的随机范围
w_random = [1600, 3000]  # 新建图的宽
h_random = [1400, 2800]  # 新建图的高
bili_list = [0.8, 1.3]  # bili > 0.8 and bili < 1.3
size_random=[0.8, 2.0] #随机resize大小的比例系数
bc_list=['./biaochi1.jpg','./biaochi2.jpg']#标尺图片
biaochi_bili=[1.2,1.6]  #标尺比例

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
    return image_rotate


def read_img(i, ImgArray,h=0):
    """
    读取图片
    :param i:第i种图
    :param ImgArray:存放读取图像
    :return:
    """
    img_box = []
    if  i==1 or i==2:
        Srcimg = cv2.imread(path + str(label2[i][h]) + '.jpg')  ##########gai label1,label2
    else:
        Srcimg = cv2.imread(path + str(label2[i]) + '.jpg')  ##########gai label1,label2

    m=random.randint(0,2)
    # print(m)
    if m==0:
        img=Srcimg
    elif m==1:
        img=morphologyImage(Srcimg)
    else:
        img=GaussProcess(Srcimg)
    # cv2.imshow("11",img)
    # cv2.waitKey(0)
    img_box.append(img)
    img_s90, = rotate_bound(img, 90)  # 顺时针旋转90
    img_box.append(img_s90)

    img_n90 = rotate_bound(img, 270)  # 逆时针旋转90
    img_box.append(img_n90)
    n = random.randint(0, 2)
    img_end = img_box[n]
    size = random.uniform(size_random[0],size_random[1])
    img_end = cv2.resize(img_end, None, fx=size, fy=size)
    ImgArray.append(img_end)

def xml(num,width,height,ImgArray,labelname,box):
    """
    写xml文件
    :param num: 第num个文件
    :param width: 图的宽
    :param height: 图的高
    :param ImgArray: 存放图的list
    :param labelname: 贴的图的名字
    :param box: 贴图的框坐标
    :return: 写好的xml文件
    """
    xml_file = open('./result2/res1_%d.xml' % num, 'w')  ######################gai res1,res2
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>IMage</folder>\n')
    xml_file.write('    <filename>' + 'res.jpg' + '</filename>\n')
    xml_file.write('    <path>' + './' + '</path>\n')
    xml_file.write('    <source>\n')
    xml_file.write('        <database>' + 'Unknown' + '</database>\n')
    xml_file.write('    </source>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>1</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('    <segmented>0</segmented>\n')
    for i in range(len(ImgArray)):
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(labelname[i]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(box[i][0][1]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(box[i][0][0]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(box[i][1][1]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(box[i][1][0]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    return xml_file

#generator json file
def generatorJson():
    pass


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
    ###############################gai#######
    for i in range(6):
        mclasses = random.randint(0, 12)
        # print(mclasses)
        if mclasses == 1 or mclasses == 2:
            lable_num = random.randint(1, 4)
            for ii in range(lable_num):
                h = random.randint(0, 2)
                read_img(mclasses, ImgArray, h)
                labelname.append(label2[mclasses][0])
        else:
            lable_num = random.randint(1, 3)  # 一张图上画的张数 随机
            for ii in range(lable_num):
                read_img(mclasses, ImgArray)
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
                #print('no ok')
                flag = True
                break
            else:
                flag = False
    height, width, _ = img.shape
    cv2.imwrite('./result2/res1_%d.jpg' % num, img)  ######################gai res1,res2
    ##xml
    xml_file=xml(num, width, height, ImgArray, labelname, box)


if __name__ == '__main__':
    for num in range(10):
        img_w, img_h = genrateImage()
        kuo(num, img_w, img_h)
        print('%d OK!' % num)
    print('END!')
