import collections
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=500, help="max buffer size")
args = vars(ap.parse_args())


tempLD = []
splitLD = []
bundleLD = []

orange_pts = collections.deque(maxlen=args["buffer"])
green_pts = collections.deque(maxlen=args["buffer"])
pink_pts = collections.deque(maxlen=args["buffer"])  #green의 pts
blue_pts = collections.deque(maxlen=args["buffer"])  #blue의 pts
yellow_pts = collections.deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(0)

time.sleep(2.0)

class hsvSet:
    def __init__(self):
        pass

    def hsvRange(self, frameset, ptsset, hsvset, hsvLowerset, hsvUpperset, tempLDset, linecolorset):
        self.hsv = hsvset
        self.hsvLower = hsvLowerset
        self.hsvUpper = hsvUpperset
        self.tempLD = tempLDset
        self.frame = frameset
        self.pts = ptsset
        self.linecolor = linecolorset

        if len(self.tempLD) > 7500:
            del(self.tempLD[-1])
            del(self.tempLD[-1])
            del(self.tempLD[-1])

        mask = cv2.inRange(self.hsv, self.hsvLower, self.hsvUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)  # 물체 제외하고 나머진 검은 배경으로

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)  # 윤곽
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # 중심 모멘트
            if radius > 3:
                cv2.circle(self.frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)  # 물체 감싸는 원
                cv2.circle(self.frame, center, 5, (0, 0, 255), -1)  # 중심점 원
        if center == None:
            self.tempLD.insert(0, -1)
            self.tempLD.insert(0, -1)
            self.tempLD.insert(0, -1)
        else:
            self.tempLD.insert(0, int(M["m10"] / M["m00"]))
            self.tempLD.insert(0, int(M["m01"] / M["m00"]))
            self.tempLD.insert(0, int(radius))

        self.pts.appendleft(center)  # 중심점 리스트
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(self.frame, self.pts[i - 1], self.pts[i], self.linecolor, thickness)

        return self.tempLD

H = hsvSet()  # 클래스 대입(이거 안 하고 바로 hsvSet.hsvRange 쓰면 오류남)
colorDic = {'orangeL':(14,95,178),'orangeU':(20,255,255),'orange':(0,128,255),
            'greenL':(32,56,78),'greenU':(43,255,255),'green':(0,255,0),
            'pinkL':(111 ,63 ,141),'pinkU':(179,255,255),'pink':(255,100,255),
            'blueL':(96,42,137),'blueU':(107,255,255),'blue':(255,0,0),
            'yellowL':(20,80,160),'yellowU':(31,255,255),'yellow':(0,255,255)}
Ntrain = int(input("train 데이터 개수 설정"))
Ntest = int(input("test 데이터 개수 설정"))
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video",False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # < orange 인식 >
    # 딕셔너리에서 HSV범위, 선 색의 값을 받아옴
    hsvLower = colorDic.get('orangeL')  # Lower은 색상+L
    hsvUpper = colorDic.get('orangeU')  # Upper은 색상+U
    linecolor = colorDic.get('orange')  # line은 색상
    tempLD = H.hsvRange(frame, orange_pts, hsv, hsvLower, hsvUpper, tempLD, linecolor)  # 메인 시스템(추적, 중심모멘트)

    # < green 인식 >
    # 딕셔너리에서 HSV범위, 선 색의 값을 받아옴
    hsvLower = colorDic.get('greenL')  # Lower은 색상+L
    hsvUpper = colorDic.get('greenU')  # Upper은 색상+U
    linecolor = colorDic.get('green')   # line은 색상
    tempLD2 = H.hsvRange(frame, green_pts, hsv, hsvLower, hsvUpper, tempLD, linecolor)  # 메인 시스템(추적, 중심모멘트)

    # < pink 인식 >
    # 딕셔너리에서 HSV범위, 선 색의 값을 받아옴
    hsvLower = colorDic.get('pinkL')  # Lower은 색상+L
    hsvUpper = colorDic.get('pinkU')  # Upper은 색상+U
    linecolor = colorDic.get('pink')  # line은 색상
    tempLD3 = H.hsvRange(frame, pink_pts, hsv, hsvLower, hsvUpper, tempLD, linecolor)  # 메인 시스템(추적, 중심모멘트)

    # < blue 인식 >
    # 딕셔너리에서 HSV범위, 선 색의 값을 받아옴
    hsvLower = colorDic.get('blueL')  # Lower은 색상+L
    hsvUpper = colorDic.get('blueU')  # Upper은 색상+U
    linecolor = colorDic.get('blue')   # line은 색상
    tempLD4 = H.hsvRange(frame, blue_pts, hsv, hsvLower, hsvUpper, tempLD, linecolor)  # 메인 시스템(추적, 중심모멘트)

    # < yellow 인식 >
    # 딕셔너리에서 HSV범위, 선 색의 값을 받아옴
    hsvLower = colorDic.get('yellowL')  # Lower은 색상+L
    hsvUpper = colorDic.get('yellowU')  # Upper은 색상+U
    linecolor = colorDic.get('yellow')  # line은 색상
    tempLD5 = H.hsvRange(frame, yellow_pts, hsv, hsvLower, hsvUpper, tempLD, linecolor)  # 메인 시스템(추적, 중심모멘트)

    # < 가공된 프레임 표시 >
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    # < 데이터 추출 >
    if key == ord("q"):
        del(tempLD[0])
        del(tempLD[0])
        del(tempLD[0])
        print('연속된 데이터: ',tempLD)
        print(len(tempLD))
        for i in range(0,len(tempLD),len(tempLD)//30):  # 30개 분할
            splitLD.append(tempLD[i])
            if len((splitLD)) == 30:  # 30으로 나눠도 31 33처럼 나오는 경우가 있어서 길이 30이면 break
                splitLD.insert(0,0)  # insert(index, 정답레이블) eg., 세모는 0 원은 1로 설정(더 많아지면 001 010 ...)
                                     # csv는 정답레이블, 좌표, 좌표, 좌표,,,,형식으로 저장
                break
        bundleLD.append(splitLD)  # 30묶음을 학습데이터에 저장
        print('분할된 데이터: ',splitLD)
        pts = collections.deque(maxlen=args["buffer"])  # pts 초기화
        tempLD = []  # tempLD 초기화
        splitLD = []  # splitLD 초기화

        # < 데이터 가공 >
        import csv
        if len(bundleLD) == (Ntrain+Ntest):  # train과 test 데이터가 모두 모이면 가공 시작
            # Ntrain개수만큼은 train 데이터
            with open('세모 데이터(train).csv', 'w', newline='') as f:
                for i in range(0,Ntrain):
                    writer = csv.writer(f)
                    writer.writerow(bundleLD[i])

            # Ntest개수만큼은 test 데이터
            with open('세모 데이터(test).csv', 'w', newline='') as f:
                for i in range(Ntrain,(Ntrain+Ntest)):
                    writer = csv.writer(f)
                    writer.writerow(bundleLD[i])
            break

if not args.get("video",False):
    vs.stop()
else:
    vs.release()


cv2.destroyAllWindows()