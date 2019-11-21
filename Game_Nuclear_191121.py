import numpy as np
import cv2
from mss import mss
import pandas as pd
import os
import sys
from PIL import Image, ImageFont, ImageDraw
from pexpect import popen_spawn
import time # time 라이브러리
# 배경화면 가져오기
winwidth = 1280 # 가로 윈도우
winheight = 800 # 세로 윈도우
prevTime = 0
#rows = 768 #행 세로 카메라
#cols = 1024 #열 가로 카메라
# 본컴터용
#bbox = {'top': 150, 'left': 350, 'width': 700, 'height': 784}
#실험용
bbox = {'top': 93, 'left': 0, 'width': winwidth, 'height': winheight}
sct = mss()

# korea coco.names coco_data.txt
with open("C:/Game_Nuclear/Shoot/coco_korea.names", "r", encoding='utf-8') as f:
    coco_labels = f.readlines()


def darknet(message):
    # os.chdir("C:/darknet-master/darknet-master/build/darknet/x64")
    os.chdir("C:/Game_Nuclear/darknet-master-prn/darknet-master/build/darknet/x64")

    # process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/%02d.jpg', yolo_im)
    # webcam용
    # process = popen_spawn.PopenSpawn('darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights')
    # 사진용
    # process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show -ext_output -save_labels')
    # yolov3
    #process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights \
    #                                -thresh 0.4 -dont_show -ext_output -save_labels')
    # yolov3-tiny
    # process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights \
    #                                -dont_show -ext_output -save_labels')
    # yolov3-tiny-prn
    process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3-tiny-prn.cfg yolov3-tiny-prn.weights \
                                     -thresh 0.2 -dont_show -ext_output -save_labels')
    # yolov3-Enet
    #process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/enet-coco.cfg  enetb0-coco_final.weights \
    #                                -dont_show -ext_output -save_labels')
    # yolov3-tiny Json 실험 중
    # process = popen_spawn.PopenSpawn('darknet.exe detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights \
    #                                -dont_show -ext_output -save_labels -out result.json')

    print(message)
    return process


message = 'Darknet Started'
darknet_process = darknet(message)

# 영상저장하기
#cap = cv2.VideoCapture(0) # 0번 카메라
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#print('frame_size =', frame_size)
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ('D', 'I', 'V', 'X')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out1 = cv2.VideoWriter('C:/Users/TaeMin Lee/Desktop/test/record0.mp4', fourcc, 20.0, (700, 784)) # 위에 bbox와 같게 할 것
#out2 = cv2.VideoWriter('C:/Users/TaeMin Lee/Desktop/test/record1.mp4', fourcc, 20.0, frame_size)
while 1:
    # ==============================================================================
    # FPS 측정을 위한 프로그램
    # 현재 시간 가져오기 (초단위로 가져옴)

    #global fps
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    #print('sec= ', sec)            
    if sec != 0 : # 1 / time per frame
        fps = 1 / (sec)
        print('fps=', fps)
    # ==============================================================================
    # 배경화면 가져오기 sct_img
    sct_img = np.array(sct.grab(bbox))
    #print(sct_img) # 행렬 확인용
    #print(len(sct_img))
    # rgb로 변경해주기 mss가 rgb로 받아오기 때문에 마지막 필요없음
    #frame1 = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
    # RGB 분해 블루0 마지막채널 제외하고 결합
    #frame1 = cv2.split(sct_img)
    #frame1 = cv2.merge([frame1[0]*0, frame1[1], frame1[2]])
    #out1.write(frame1)
    frame1 = sct_img
    #print(frame1)
    #print(len(frame1))
    #print(frame1.shape)
    #cv2.imshow('screen', frame1)
    # 영상저장하기
    #retval, frame2 = cap.read()
    #if not retval:
    #    break
    #out2.write(frame2)
    #cv2.imshow('frame2', frame2)
    # 화면리사이징
    #frame2 = cv2.resize(frame2, (cols, rows), interpolation=cv2.INTER_LINEAR)
    #cv2.imshow('reeframe2', frame2)
    '''.
# 어파인 변환 패어스팩티브
    rows, cols, channels = frame2.shape
    pts1 = np.float32([[cols*4/10, rows*3/10], [cols*6/10, rows*3/10],
                       [cols*1/10, rows*8/10], [cols*9/10, rows*8/10]])
    pts2 = np.float32([[0, 0], [winwidth, 0],
                       [0, winheight], [winwidth, winheight]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #frame1 = cv2.warpAffine(frame1, M, (rows, cols))
    reframe2 = cv2.warpPerspective(frame2, M, (winwidth, winheight))
    #cv2.imshow('topview', reframe2)

# 영상 합성
    src1 = reframe2
    src2 = frame1
    #cv2.imshow('src2', src2)
    # 1
    f1_rows, f1_cols, f1_channels = src1.shape
    f2_rows, f2_cols, f2_channels = src2.shape
    roi = src1
    #roi = src1[(f1_rows-f2_rows)//2:(f1_rows-f2_rows)//2+f2_rows,
    #           (f1_cols-f2_cols)//2:(f1_cols-f2_cols)//2+f2_cols]
    # 2
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    #cv2.imshow('mask', mask)
    #cv2.imshow('mask_inv', mask_inv)
    # 3
    src1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    #src1_bg = cv2.add(roi, roi, mask=mask) #이거하면 밝아짐 안됨
    #cv2.imshow('src1_bg', src1_bg)
    # 4
    src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv)
    #cv2.imshow('src2_fg', src2_fg)
    # 5
    ##dst = cv2.add(src1_bg, src2_fg)
    dst = cv2.bitwise_or(src1_bg, src2_fg)
    #cv2.imshow('dst', dst)
    # 6
    # 최종본 자동으로 가져오기
    np.copyto(roi, dst)
    # 최종본 지정해서 넣기
    #src1[(f1_rows-f2_rows)//2:(f1_rows-f2_rows)//2+f2_rows,
    #     (f1_cols-f2_cols)//2:(f1_cols-f2_cols)//2+f2_cols] = dst
    #out2.write(src1)
    cv2.imshow('src1', src1)

    # 역어파인 변환 패어스팩티브
    toprows, topcols, topchannels = src1.shape
    pts1 = np.float32([[0, 0], [winwidth, 0],
                       [0, winheight], [winwidth, winheight]])
    pts2 = np.float32([[cols*4/10, rows*3/10], [cols*6/10, rows*3/10],
                       [cols*1/10, rows*8/10], [cols*9/10, rows*8/10]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # frame1 = cv2.warpAffine(frame1, M, (rows, cols))
    re3dframe2 = cv2.warpPerspective(src1, M, (cols, rows), flags=cv2.INTER_LINEAR)
    cv2.imshow('3d', re3dframe2)

    # 영상 합성
    src11 = frame2
    src12 = re3dframe2
    # cv2.imshow('src2', src2)
    # 1
    #f11_rows, f11_cols, f11_channels = src11.shape
    #f12_rows, f12_cols, f12_channels = src12.shape
    roi2 = src11
    #roi2 = src11[(f11_rows-f12_rows)//2:(f11_rows-f12_rows)//2+f12_rows,
    #             (f11_cols-f12_cols)//2:(f11_cols-f12_cols)//2+f12_cols]
    # 2
    gray2 = cv2.cvtColor(src12, cv2.COLOR_BGR2GRAY)
    ret2, mask2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY_INV)
    mask2_inv = cv2.bitwise_not(mask2)
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask_inv', mask_inv)
    # 3
    src11_bg = cv2.bitwise_and(roi2, roi2, mask=mask2)
    # src1_bg = cv2.add(roi, roi, mask=mask) #이거하면 밝아짐 안됨
    # cv2.imshow('src1_bg', src1_bg)
    # 4
    src12_fg = cv2.bitwise_and(src12, src12, mask=mask2_inv)
    # cv2.imshow('src2_fg', src2_fg)
    # 5
    ##dst = cv2.add(src1_bg, src2_fg)
    dst2 = cv2.bitwise_or(src11_bg, src12_fg)
    # cv2.imshow('dst', dst)
    # 6
    # 최종본 자동으로 가져오기
    np.copyto(roi2, dst2)
    # 최종본 지정해서 넣기
    # src1[(f1_rows-f2_rows)//2:(f1_rows-f2_rows)//2+f2_rows,
    #     (f1_cols-f2_cols)//2:(f1_cols-f2_cols)//2+f2_cols] = dst
    #out2.write(src11)
    cv2.imshow('src11', src11)

    #행렬보기
    fe = cv2.split(frame2)
    df = pd.DataFrame(fe[0]+fe[1]+fe[2])
    print(df)
    '''

# ==============================================================================
    # 이미지 저장
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    carla_scene_data = Image.fromarray(frame1, 'RGB')
    carla_scene_data.save('C:/Game_Nuclear/Shoot/Yolo_image.jpg')
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    # ==============================================================================
    #global darknet_process
    carla_scene_exists = os.path.isfile("C:/Game_Nuclear/Shoot/Yolo_image.jpg")
    if carla_scene_exists:
        carla_scene = (b"C:/Game_Nuclear/Shoot/Yolo_image.jpg")
        darknet_process.send(carla_scene + b'\n')
        labels_file_exists = os.path.isfile("C:/Game_Nuclear/Shoot/Yolo_image.txt")
        if labels_file_exists:
            labels_file = open("C:/Game_Nuclear/Shoot/Yolo_image.txt", "r")
            predicted_labels = labels_file.read()
            if os.stat("C:/Game_Nuclear/Shoot/Yolo_image.txt").st_size > 10:  # !=0
                # ==============================================================================
                # 라벨 위치에 따른 명칭 가져오기
                labels = np.mat(predicted_labels)
                labels_shape = labels.shape
                labels_row = labels_shape[1] // 5
                labels_Mat = labels.reshape((labels_row, 5))
                # ==============================================================================
                for label_i in range(0, labels_row):

                    sign_tsr = labels_Mat[label_i, 0]
                    #global coco_labels
                    # ==============================================================================
                    # 라벨에 따라 색상변화 프로그램
                    #boxColor = (int(255 * (1 - (sign_tsr ** 2))), int(255 * (sign_tsr ** 2)), 0)
                    if sign_tsr == 0 :
                        boxColor = (0, 0, 255)
                    elif sign_tsr == 1 or sign_tsr == 2 or sign_tsr == 3 or sign_tsr == 5 or sign_tsr == 7:
                        boxColor = (0, 255, 0)
                    else : boxColor = (int(255 * (1 - (sign_tsr ** 2))), int(255 * (sign_tsr ** 2)), int(255 * (sign_tsr ** 2)))
                    # ==============================================================================
                    # 중심값을 이용한 물체 라운딩 작업
                    cv2.rectangle(frame1,
                                  (int((winwidth * labels_Mat[label_i, 1]) - (winwidth * labels_Mat[label_i, 3] // 2)),
                                   int((winheight * labels_Mat[label_i, 2]) - (winheight * labels_Mat[label_i, 4] // 2))),
                                  (int((winwidth * labels_Mat[label_i, 1]) + (winwidth * labels_Mat[label_i, 3] // 2)),
                                   int((winheight * labels_Mat[label_i, 2]) + (winheight * labels_Mat[label_i, 4] // 2))),
                                  boxColor, 2)
                    # ==============================================================================
                    # detection line
                    # 자동차와 관련된 것들만 선을 만들도록 함(파란색 선으로 검출된 것을 표시함)
                    # 0->사람 / 1->자전거 / 2->승용차 / 3->오토바이 / 5->버스 / 7->트럭
                    if sign_tsr == 0 :
                        cv2.line(frame1, (int(winwidth * labels_Mat[label_i, 1]), int(winheight * labels_Mat[label_i, 2])),
                                 (winwidth//2, winheight), (0, 0, 255), 3)
                    elif sign_tsr == 1 or sign_tsr == 2 or sign_tsr == 3 or sign_tsr == 5 or sign_tsr == 7:
                        cv2.line(frame1, (int(winwidth * labels_Mat[label_i, 1]), int(winheight * labels_Mat[label_i, 2])),
                                 (winwidth//2, winheight), (0, 255, 0), 3)
                    else : cv2.line(frame1, (int(winwidth * labels_Mat[label_i, 1]), int(winheight * labels_Mat[label_i, 2])),
                                 (winwidth//2, winheight), (255, 0, 0), 3)
                    # ==============================================================================
                    # detection circle
                    cx = int(winwidth * labels_Mat[label_i, 1])
                    cy = int(winheight * labels_Mat[label_i, 2])
                    cv2.circle(frame1, (cx, cy), 2, (0, 0, 255), -1) # yolo v3
                    # ==============================================================================
                    # Text File read code
                    # 영어로 검출된 물체의 라벨을 표시함
                    '''
                    #org = (int((640 * labels_Mat[label_i, 1]) - (640 * labels_Mat[label_i, 3] // 2)), int(-5+(480 * labels_Mat[label_i, 2]) - (480 * labels_Mat[label_i, 4] // 2)))
                    org = (int((640 * labels_Mat[label_i, 1]) - (640 * labels_Mat[label_i, 3] // 2)),
                           int((480 * labels_Mat[label_i, 2]) - (480 * labels_Mat[label_i, 4] // 2)))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    size, baseLine = cv2.getTextSize(coco_labels[sign_tsr], font, 1, 2)
                    cv2.rectangle(size_im, org, (org[0] + size[0], org[1] - size[1]), boxColor)
                    cv2.putText(size_im, coco_labels[sign_tsr], org, font, 1, (255,255,255), 2)
                    '''
                    # 한글로 검출된 물체의 라벨을 표시함
                    # unicode_font = ImageFont.truetype(font = "NanumGothic.ttf", size = 20)
                    # size_im.coco_labels[sign_tsr](org, coco_labels[sign_tsr], (255,255,255), font = unicode_font)
                    org_korea = (int((winwidth * labels_Mat[label_i, 1]) - (winwidth * labels_Mat[label_i, 3] // 2)),
                                 int(-16 + (winheight * labels_Mat[label_i, 2]) - (winheight * labels_Mat[label_i, 4] // 2)))
                    font_path = "C:/Game_Nuclear/Shoot/NanumGothic.ttf"
                    font = ImageFont.truetype(font_path, 16)
                    img_pil = Image.fromarray(frame1)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(org_korea, coco_labels[int(sign_tsr)], font=font, fill=(255, 255, 255))
                    frame1 = np.array(img_pil)
                    # ==============================================================================
    cv2.imshow('bat', frame1)
                        #종료 ESC 누르기
    key = cv2.waitKey(25)
    if key == 27:
        break
cap.release()
#out1.release()
#out2.release()
cv2.destroyAllWindows()