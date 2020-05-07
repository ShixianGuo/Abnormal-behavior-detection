#coding=utf-8
import cv2

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    # 如果符合条件，返回True，否则返回False
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 根据坐标画出人物所在的位置
def draw_person(img, person):
  x, y, w, h = person
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


# 图片识别方法
def discern(img):
    # 定义HOG特征+SVM分类器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)

    # 判断坐标位置是否有重叠
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            a = is_inside(r, q)
            if ri != qi and a:
                  break
            else:
                  found_filtered.append(r)
                
    # 勾画筛选后的坐标位置
    for person in found_filtered:
        draw_person(img, person)


    #人脸检测
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier(
        "C:\Program Files\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
    )
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        grayImg, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)

    cv2.imshow("image", img)  # 显示图像


def main():  
    cap = cv2.VideoCapture(0)

    while (1):
        ret, frame = cap.read()
        discern(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
