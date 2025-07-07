# 이미지 변환 (팽창 Dilation)
import cv2
import numpy as np


kernel = np.ones((3, 3), dtype=np.uint8)
# kernel

img = cv2.imread('./dilate.png', cv2.IMREAD_GRAYSCALE)
dilate1 = cv2.dilate(img, kernel, iterations=1) # 반복 횟수
dilate2 = cv2.dilate(img, kernel, iterations=2)
dilate3 = cv2.dilate(img, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('dilate1', dilate1)
cv2.imshow('dilate2', dilate2)
cv2.imshow('dilate3', dilate3)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 변환 (침식) - 이미지를 깎아서 노이즈 제거
import cv2
import numpy as np

kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('./erode.png', cv2.IMREAD_GRAYSCALE)
erode1 = cv2.erode(img, kernel, iterations=1) # 1회 반복
erode2 = cv2.erode(img, kernel, iterations=2)
erode3 = cv2.erode(img, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('erode1', erode1)
cv2.imshow('erode2', erode2)
cv2.imshow('erode3', erode3)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 변환 
# 열림(Opening) - 침식 후 팽창
import cv2
import numpy as np

kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('./erode.png', cv2.IMREAD_GRAYSCALE)

erode = cv2.erode(img, kernel, iterations=3)
dilate = cv2.dilate(erode, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('dilat', dilate)
cv2.imshow('erode', erode)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 닫힘 (Closing) : 팽창 후 침식
import cv2
import numpy as np

kernel = np.ones((3, 3), dtype=np.uint8)

img = cv2.imread('./dilate.png', cv2.IMREAD_GRAYSCALE)

dilate = cv2.dilate(img, kernel, iterations=3)
erode = cv2.erode(dilate, kernel, iterations=3)

cv2.imshow('img', img)
cv2.imshow('dilat', dilate)
cv2.imshow('erode', erode)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 검출 (경계선)
# Canny Edge Detection - 캐니 알고리즘
import cv2

img = cv2.imread('./snowman.png')

canny = cv2.Canny(img, 150, 200)
# 대상 이미지, minVal (하위임계값), maxVal (상위임계값)

cv2.imshow('img', img)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Canny Edge Detection - 캐니 알고리즘 - 트랙바 적용
# 로컬 pc에서 실행할 때
import cv2

def empty(pos):
    pass

img = cv2.imread('./snowman.png')

name = "Trackbar"
cv2.namedWindow(name)
cv2.createTrackbar('threshold1', name, 0, 255, empty) # minVal
cv2.createTrackbar('threshold2', name, 0, 255, empty) # maxVal

while True:
    threshold1 = cv2.getTrackbarPos('threshold1', name)
    threshold2 = cv2.getTrackbarPos('threshold2', name)

    canny = cv2.Canny(img, threshold1, threshold2)
    # 대상 이미지, minVal (하위임계값), maxVal (상위임계값)

    cv2.imshow('img', img)
    cv2.imshow('canny', canny)

    if cv2.waitKey(1) == ord('q'):
         break

cv2.destroyAllWindows()




# 이미지 검출(윤곽선)
import cv2

img = cv2.imread('./card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # 윤곽선 검출
# 윤곽선 정보, 계층 구조
# 이미지, 윤곽선 찾는 모드 (mode), 윤곽선 찾을때 사용하는 근사치 방법 (method) : CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE

COLOR = (0, 200, 0) # 녹색
cv2.drawContours(target_img, contours, -1, COLOR, 2) # 윤곽선 그리기
# 대상 이미지, 윤곽선 정보, 인덱스 (-1 이면 전체), 색깔, 두께

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('target_img', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 윤곽선 찾기 모드 - 외곽 윤곽선
import cv2

img = cv2.imread('./card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(hierarchy)
# print(f'총 발견 갯수 : {len(contours)}')

COLOR = (0, 200, 0) # 녹색
cv2.drawContours(target_img, contours, -1, COLOR, 2)

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('target_img', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 경계 사각형(바운딩박스)
import cv2

img = cv2.imread('./card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0) # 녹색

for cnt in contours:
    x, y, width, height = cv2.boundingRect(cnt)
    cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2) # 사각형 그림

cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('otsu', otsu)
cv2.imshow('target_img', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 면적
import cv2

img = cv2.imread('./card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0) # 녹색

for cnt in contours:
    if cv2.contourArea(cnt) > 25000:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2) # 사각형 그림

cv2.imshow('img', img)
cv2.imshow('target_img', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 스스로 해보기 : 개별 카드 추출해서 파일 저장
import cv2

img = cv2.imread('./card.png')
target_img = img.copy() # 사본 이미지

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

COLOR = (0, 200, 0) # 녹색

idx = 1

for cnt in contours:   # 윤곽선의 개수를 순서대로 돌려라
    if cv2.contourArea(cnt) > 25000:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2) # 사각형 그림

        # 직사각형 영역에 해당하는 이미지만 자르기
        crop = img[y:y+height, x:x+width]
        cv2.imshow('crop', crop)
        # 자른 이미지 저장하기
        cv2.imwrite(f'crop_card_{idx}.png', crop)
        idx += 1

cv2.imshow('img', img)
cv2.imshow('target_img', target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




# 11. 퀴즈
# 회전 : 시계 반대방향으로 90도
# 재생속도 (FPS) : 원본 x 4배
# 출력 파일명 : city_output.avi (코덱 : DIVX)
# 원본 파일명 : city.mp4
# 예제 동영상 : https://www.pexels.com/video/3121459/
# 크기 : HD (1280 x 720)
# 파일명 : city.mp4

import cv2

cap = cv2.VideoCapture('./city.mp4')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('city_output.avi', fourcc, fps*4, (height, width))   # 비디오 객체 생성

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rotate_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)   # 시계 반대방향으로 90도
    out.write(rotate_frame)

    # 확인
    cv2.imshow('video', frame)
    cv2.imshow('output', rotate_frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()




# 프로젝트 : 얼굴을 인식하여 캐릭터 씌우기
# pip install mediapipe
import cv2
import mediapipe as mp

def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
    alpha = overlay_image[:, :, 3] # BGRA
    mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
    # (255, 255)  ->  (1, 1)
    # (255, 0)        (1, 0)
    
    # 1 - mask_image ?
    # (0, 0)
    # (0, 1)
    
    for c in range(0, 3): # channel BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))

# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 drawing_utils 모듈을 사용

# 동영상 파일 열기
cap = cv2.VideoCapture('./face_video2.mp4')

# 이미지 불러오기
image_right_eye = cv2.imread('./img/right_eye4.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_left_eye = cv2.imread('./img/left_eye4.png', cv2.IMREAD_UNCHANGED) # 100 x 100
image_nose = cv2.imread('./img/nose4.png', cv2.IMREAD_UNCHANGED) # 300 x 100 (가로, 세로)

# 임계치 0.7
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR은 opencv 읽기 방식, RGB는 Mediapipe 읽기 방식
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            # 6개 특징 : 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀 (귀구슬점, 이주)
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                # print(detection)
                
                # 특정 위치 가져오기
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0] # 오른쪽 눈
                left_eye = keypoints[1] # 왼쪽 눈
                nose_tip = keypoints[2] # 코 끝부분
                
                h, w, _ = image.shape # height, width, channel : 이미지로부터 세로, 가로 크기 가져옴
                right_eye = (int(right_eye.x * w) - 20, int(right_eye.y * h) - 150) # 이미지 내에서 실제 좌표 (x, y) -
                left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 150)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))
                
                # (디버깅용) 양 눈에 동그라미 그리기
                #cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA) # 파란색
                #cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA) # 초록색                
                # 코에 동그라미 그리기
                #cv2.circle(image, nose_tip, 55, (0, 255, 255), 10, cv2.LINE_AA) # 노란색
                

                # image, x, y, w, h, overlay_image
                overlay(image, *right_eye, 50, 50, image_right_eye)
                overlay(image, *left_eye, 50, 50, image_left_eye)
                overlay(image, *nose_tip, 150, 50, image_nose)
                
    
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))
        
        if cv2.waitKey(1) == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()