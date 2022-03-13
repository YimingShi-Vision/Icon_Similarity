import cv2
import numpy as np
from PIL import Image
import math


def fetchhierarchy(path):

    img1 = Image.open(path)
    width, height = img1.size
    height = height + 10
    width = width + 10
    background = Image.new("RGB", (width, height), "white")
    background.paste(img1, (5, 5))
    background.save('phone/temp/resized.png', 'PNG')

    img = cv2.imread('phone/temp/resized.png', 0)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 127, 255, 0)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(np.size(cnts))   # 得到该图中总的轮廓数量
    # print(cnts[3])    # 打印出第一个轮廓的所有点的坐标， 更改此处的0，为0--（总轮廓数-1），可打印出相应轮廓所有点的坐标
    # print(hierarchy)  # 打印出相应轮廓之间的关系

    # 对轮廓层级关系进行编号标记
    hienum = []
    i = 0
    while i < np.size(cnts):
        hienum.insert(i, -1)
        i += 1

    i = 0
    while i < np.size(cnts):
        if hierarchy[0][i][3] == 0:
            hienum[i] = 0
        i += 1

    flag = 1  # 判断是否存在下一级轮廓
    j = -1   # 当前级数编号
    while flag == 1:
        flag = 0
        j += 1
        i = 0
        while i < np.size(cnts):
            if hienum[i] == j:
                if hierarchy[0][i][2] != -1:
                    flag = 1
                    hienum[hierarchy[0][i][2]] = j+1
            i += 1

    # print(hienum)
    return cnts, hienum


def fetchcounts(hienum, m):
    counts = []
    hie = 0
    while hie <= m:
        c = 0
        i = 0
        while i < len(hienum):
            if hienum[i] == hie:
                c += 1
            i += 1
        counts.insert(hie, c)
        hie += 1
    return counts


def calculateareasame(cnts, hienum, m):
    areas = []
    totalarea = []
    hie = 0  # 当前层级数
    while hie <= m:
        area = []
        total = 0
        i = 0
        while i < len(hienum):
            area.insert(i, 0)
            i += 1
        i = 0
        while i < len(hienum):
            if hienum[i] == hie:
                area[i] = cv2.contourArea(cnts[i])
                total = total + area[i]
                # print(total)
            i += 1
        i = 0
        while i < len(hienum):
            if hienum[i] == hie:
                area[i] = area[i] / total
            i += 1
        areas.insert(hie, area)
        totalarea.insert(hie, total)
        hie += 1
    return areas, totalarea


def imageseg(cnt1, cnt2):
    rect1 = cv2.boundingRect(cnt1)
    rect2 = cv2.boundingRect(cnt2)
    if rect1[2] > rect2[2]:
        wmax = rect1[2]
    else:
        wmax = rect2[2]
    if rect1[3] > rect2[3]:
        hmax = rect1[3]
    else:
        hmax = rect2[3]
    backgroundis = Image.new("RGB", (wmax, hmax), "white")
    backgroundis.save('phone/temp/ImgSegBG.png', 'PNG')

    backgroundis = cv2.imread('phone/temp/ImgSegBG.png', 0)
    cv2.drawContours(backgroundis, cnt1, -1, (0, 0, 0), thickness=5, offset=(-rect1[0], -rect1[1]))
    cv2.imwrite('phone/temp/ImgSeg1.png', backgroundis)

    backgroundis = cv2.imread('phone/temp/ImgSegBG.png', 0)
    cv2.drawContours(backgroundis, cnt2, -1, (0, 0, 0), thickness=5, offset=(-rect2[0], -rect2[1]))
    cv2.imwrite('phone/temp/ImgSeg2.png', backgroundis)


def dhash():
    im1 = Image.open('phone/temp/ImgSeg1.png')
    im2 = Image.open('phone/temp/ImgSeg2.png')
    p = 15  # hash_value

    def cut_image(image, hash_size):
        # 将图像缩小成9*8并转化成灰度图,（可以改）这里是16*15
        image_1 = image.resize((hash_size + 1, hash_size), Image.ANTIALIAS).convert('L')
        pixel = list(image_1.getdata())
        return pixel

    def trans_hash(lists):
        # 比较列表中相邻元素大小
        j = len(lists) - 1
        hash_list = []
        m, n = 0, 1
        for i in range(j):
            if lists[m] > lists[n]:
                hash_list.append(1)
            else:
                hash_list.append(0)
            m += 1
            n += 1
        return hash_list

    def difference_value(image_lists):
        # 获取图像差异值并获取指纹
        assert len(image_lists) == p * (p + 1), "size error"
        m, n = 0, p + 1
        hash_list = []
        for i in range(0, p):
            slc = slice(m, n)
            image_slc = image_lists[slc]
            hash_list.append(trans_hash(image_slc))
            m += p + 1
            n += p + 1
        return hash_list

    def calc_distance(image1, image2):
        # 计算汉明距离
        image1_lists = cut_image(image1, p)
        image2_lists = cut_image(image2, p)
        hash_lists1 = difference_value(image1_lists)
        hash_lists2 = difference_value(image2_lists)
        calc = abs(np.array(hash_lists2) - np.array(hash_lists1))
        return calc

    calc1 = calc_distance(im1, im2)
    simi = 1.00 - np.sum(calc1) / (p * p)
    # print('Similarity=' + str(simi * 100) + '%')
    return simi


def compare_same_dhash(cnt, hie, c, area, total, hienum):
    if c[0][hienum] == 0 or c[1][hienum] == 0:
        if c[0][hienum] != 0:
            return 0, 0
        else:
            return 0, 1
    else:
        if c[0][hienum] > c[1][hienum]:
            base = 0
            comp = 1
        elif c[0][hienum] < c[1][hienum]:
            base = 1
            comp = 0
        else:
            if total[0][hienum] > total[1][hienum]:
                base = 0
                comp = 1
            else:
                base = 1
                comp = 0
        # 本函数核心
        simimaxes = []
        i = 0
        while i < np.size(cnt[base]):
            simimaxes.insert(i, 0)
            i += 1
        i = 0
        while i < np.size(cnt[base]):
            if hie[base][i] == hienum:
                j = 0
                while j < np.size(cnt[comp]):
                    if hie[comp][j] == hienum:
                        imageseg(cnt[base][i], cnt[comp][j])
                        simi = dhash()
                        if simi > simimaxes[i]:
                            simimaxes[i] = simi
                    j += 1
            i += 1
        simifinal = 0
        i = 0
        while i < np.size(cnt[base]):
            simifinal = simifinal + simimaxes[i] * area[base][hienum][i]
            i += 1
        return simifinal, base


def compare_same_hu(cnt, hie, c, area, total, hienum):
    if c[0][hienum] == 0 or c[1][hienum] == 0:
        if c[0][hienum] != 0:
            return 0, 0
        else:
            return 0, 1
    else:
        if c[0][hienum] > c[1][hienum]:
            base = 0
            comp = 1
        elif c[0][hienum] < c[1][hienum]:
            base = 1
            comp = 0
        else:
            if total[0][hienum] > total[1][hienum]:
                base = 0
                comp = 1
            else:
                base = 1
                comp = 0
        # 本函数核心
        simimaxes = []
        i = 0
        while i < np.size(cnt[base]):
            simimaxes.insert(i, 0)
            i += 1
        i = 0
        while i < np.size(cnt[base]):
            if hie[base][i] == hienum:
                j = 0
                while j < np.size(cnt[comp]):
                    if hie[comp][j] == hienum:
                        simi = cv2.matchShapes(cnt[base][i], cnt[comp][j], cv2.CONTOURS_MATCH_I2, 0.0)
                        simi = 1 - math.log(simi + 1, 11)  # 归一化
                        if simi > simimaxes[i]:
                            simimaxes[i] = simi
                    j += 1
            i += 1
        simifinal = 0
        i = 0
        while i < np.size(cnt[base]):
            simifinal = simifinal + simimaxes[i] * area[base][hienum][i]
            i += 1
        return simifinal, base


def compare_diff_dhash(cnt, hie, c, area, total, hmax):
    simi, base = compare_same_dhash(cnt, hie, c, area, total, 0)
    if hmax == 0:
        return simi
    else:
        hi = 1
        while hi <= hmax:
            simi2, base = compare_same_dhash(cnt, hie, c, area, total, hi)
            simi = (simi * ((total[base][hi-1] - total[base][hi]) / total[base][hi-1])
                    + simi2 * (total[base][hi] / total[base][hi-1]))
            hi += 1
        return simi


def compare_diff_hu(cnt, hie, c, area, total, hmax):
    simi, base = compare_same_hu(cnt, hie, c, area, total, 0)
    if hmax == 0:
        return simi
    else:
        hi = 1
        while hi <= hmax:
            simi2, base = compare_same_hu(cnt, hie, c, area, total, hi)
            simi = (simi * ((total[base][hi-1] - total[base][hi]) / total[base][hi-1])
                    + simi2 * (total[base][hi] / total[base][hi-1]))
            hi += 1
        return simi


# 主程序
filename1 = 'phone/1.jpg'
filename2 = 'phone/3-1.jpg'

cnts1, hienum1 = fetchhierarchy(filename1)
cnts2, hienum2 = fetchhierarchy(filename2)
contours = [cnts1, cnts2]
hienums = [hienum1, hienum2]
hiemax = -1
for h in hienum1:
    if h > hiemax:
        hiemax = h
for h in hienum2:
    if h > hiemax:
        hiemax = h

# print(hienum1)
# print(hienum2)
# print(hiemax)

counts1 = fetchcounts(hienum1, hiemax)
counts2 = fetchcounts(hienum2, hiemax)
countss = [counts1, counts2]

# print(counts1)
# print(counts2)

areas1, totalarea1 = calculateareasame(cnts1, hienum1, hiemax)
areas2, totalarea2 = calculateareasame(cnts2, hienum2, hiemax)
areass = [areas1, areas2]
totalareas = [totalarea1, totalarea1]

# print(areas1)
# print(totalarea1)
# print(areas2)
# print(totalarea2)

simidhash = compare_diff_dhash(contours, hienums, countss, areass, totalareas, hiemax)
simihu = compare_diff_hu(contours, hienums, countss, areass, totalareas, hiemax)
print(filename1 + ' & ' + filename2)
print('dHash: ' + str(simidhash * 100) + ' %')
print('Hu: ' + str(simihu * 100) + ' %')
