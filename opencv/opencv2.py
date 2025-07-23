# 빈 스케치북 만들기
import cv2
import numpy as np

# 세로 480 x 가로 640, 3 Channel (RGB) 에 해당하는 스케치북 만들기
img = np.zeros((480, 640, 3), dtype=np.uint8)

# img[:] = (255, 255, 255)   # 주석해제 선택: 전체 공간을 흰 색으로 채우기

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 일부 영역 색칠
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)
img[100:200, 200:300] = (255, 255, 255)   # [세로 영역, 가로 영역]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 직선
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 255)   # BGR : Yellow, 색깔
THICKNESS = 3   # 두께

# 그릴 위치, 시작 점, 끝 점, 색깔, 두께, 선 종류
cv2.line(img, (50, 100), (400, 50), COLOR, THICKNESS, cv2.LINE_8)
cv2.line(img, (50, 200), (400, 150), COLOR, THICKNESS, cv2.LINE_4)
cv2.line(img, (50, 300), (400, 250), COLOR, THICKNESS, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 원
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (255, 255, 0)   # BGR 옥색
RADIUS = 50   # 반지름
THICKNESS = 10   # 두께

# 그릴 위치, 원의 중심점, 반지름, 색깔, 두께, 선 종류
cv2.circle(img, (200, 100), RADIUS, COLOR, THICKNESS, cv2.LINE_AA)   # 속이 빈 원
cv2.circle(img, (400, 100), RADIUS, COLOR, cv2.FILLED, cv2.LINE_AA)   # 꽉 찬 원

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 사각형
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 255, 0)   # BGR 초록색
THICKNESS = 3   # 두께

# 그릴 위치, 왼쪽 위 좌표, 오른쪽 아래 좌표, 색깔, 두께
cv2.rectangle(img, (100, 100), (200, 200), COLOR, THICKNESS)   # 속이 빈 사각형
cv2.rectangle(img, (300, 100), (400, 300), COLOR, cv2.FILLED)   # 꽉 찬 사각형

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 다각형
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

COLOR = (0, 0, 255)   # BGR 빨간색
THICKNESS = 3   # 두께

pts1 = np.array([[100, 100], [200, 100], [100, 200]])
pts2 = np.array([[200, 100], [300, 100], [300, 200]])

# 그릴 위치, 그릴 좌표들, 닫힘 여부, 색깔, 두께, 선 종류
# cv2.polylines(img, [pts1], True, COLOR, THICKNESS, cv2.LINE_AA)
# cv2.polylines(img, [pts2], True, COLOR, THICKNESS, cv2.LINE_AA)
cv2.polylines(img, [pts1, pts2], True, COLOR, THICKNESS, cv2.LINE_AA) # 속이 빈 다각형

# 그릴 위치, 그릴 좌표들, 색깔, 선 종류
pts3 = np.array([[[100, 300], [200, 300], [100, 400]], [[200, 300], [300, 300], [300, 400]]])
cv2.fillPoly(img, pts3, COLOR, cv2.LINE_AA)   # 꽉 찬 다각형

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 텍스트
import cv2
import numpy as np

img = np.zeros((480, 640, 3), dtype=np.uint8)

SCALE = 1   # 크기
COLOR = (255, 255, 255)   # 흰색
THICKNESS = 1   # 두께

# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께
cv2.putText(img, "AI Simplex", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "AI Plain", (20, 150), cv2.FONT_HERSHEY_PLAIN, SCALE, COLOR, THICKNESS)
cv2.putText(img, "AI Script Simplex", (20, 250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "AI Triplex", (20, 350), cv2.FONT_HERSHEY_TRIPLEX, SCALE, COLOR, THICKNESS)
cv2.putText(img, "AI Italic", (20, 450), cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, SCALE, COLOR, THICKNESS)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 저장 - jpg
import cv2

img = cv2.imread('./img.jpg', cv2.IMREAD_GRAYSCALE)   # 흑백으로 이미지 불러오기
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('img_save.jpg', img)
print(result)




# 이미지 저장 - png
import cv2

img = cv2.imread('./opencv/img.jpg', cv2.IMREAD_GRAYSCALE)   # 흑백으로 이미지 불러오기
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img_save.png', img)   # png 형태로 저장




# 동영상 저장
import cv2

# 1) 입력 영상 경로
input_path = './video.mp4'
cap = cv2.VideoCapture(input_path)

# 2) 코덱 정의 (DIVX)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 3) 원본 프레임 크기, FPS 읽기
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) * 2   # 재생 속도를 2배로 설정

# 4) 출력 파일 설정 - 객체 생성
output_path = './video_fast.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 5) 프레임 처리 루프 - 덮어씌움
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)        # 영상 프레임만 저장 (소리 제외)
    frame = cv2.resize(frame, (360, 640)) # 출력 크기 조절
    
    cv2.imshow('frame', frame)       # 화면 출력 (옵션)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6) 리소스 해제 및 AVI 파일 다운로드
cap.release()
out.release()
# files.download(output_path)   # (코랩) 다운로드

cv2.destroyAllWindows()

print(output_path)

print('DONE')   # 정상적으로 동작하면 'DONE' 출력




# 이미지 크기 조정 - 고정 크기로 설정
import cv2

img = cv2.imread('./img.jpg')
dst = cv2.resize(img, (400, 500)) # width, height 고정 크기

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 크기 조정 - 비율로 설정
import cv2

img = cv2.imread('./img.jpg')
dst = cv2.resize(img, None, fx=0.5, fy=0.5)   # x, y 비율 정의 (0.5 배로 축소)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 보간법 - 축소
import cv2

img = cv2.imread('./img.jpg')
dst = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # x, y 비율정의 (0.5배 축소)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 보간법 - 확대
import cv2

img = cv2.imread('./img.jpg')
dst = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)   # x, y 비율정의 (1.5 배 확대)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# (jupyter 상에서 실행 X) 동영상
# 고정 크기로 설정 - ipython 상에서 터미널에 직접 입력하여 실행 가능
import cv2

cap = cv2.VideoCapture('./video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (400, 500))
    cv2.imshow('frame_resized', frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):   # 'q'키로 종료
        break

cap.release()
cv2.destroyAllWindows()




# 비율로 설정 - interpolation 적용 - ipython 상에서 터미널에 직접 입력하여 실행 가능
import cv2

cap = cv2.VideoCapture('./video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame_resized', frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):   # 'q'키로 종료
        break

cap.release()
cv2.destroyAllWindows()




# 영역을 잘라서 새로운 윈도우(창)에 표시
import cv2

img = cv2.imread('./img.jpg')
# img.shape # (390, 640, 3)

crop = img[100:200, 200:400] # 세로 기준 100 : 200 까지, 가로 기준 200 : 400 까지 자름

cv2.imshow('img', img) # 원본 이미지
cv2.imshow('crop', crop) # 잘린 이미지
cv2.waitKey(0)
cv2.destroyAllWindows()




# 영역을 잘라서 기존 윈도우에 표시
import cv2

img = cv2.imread('./img.jpg')

crop = img[100:200, 200:400] # 세로 기준 100 : 200 까지, 가로 기준 200 : 400 까지 자름
img[100:200, 400:600] = crop

cv2.imshow('img', img) # 원본 이미지
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 대칭 Flip
# 좌우대칭 - 미러반사(거울반사)
import cv2

img = cv2.imread('./img.jpg')
flip_horizontal = cv2.flip(img, 1) # flipCode > 0 : 좌우 대칭 Horizontal

cv2.imshow('img', img)
cv2.imshow('flip_horizontal', flip_horizontal)
cv2.waitKey(0)




# 상하대칭 - V-미러 반사
import cv2

img = cv2.imread('./img.jpg')
flip_vertical = cv2.flip(img, 0) # flipCode == 0 : 상하 대칭 Vertical

cv2.imshow('img', img)
cv2.imshow('flip_vertical', flip_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 상하좌우 대칭 - 180도 회전
import cv2

img = cv2.imread('./img.jpg')
flip_both = cv2.flip(img, -1) # flipCode < 0 : 상하좌우 대칭

cv2.imshow('img', img)
cv2.imshow('flip_both', flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 시계방향 90도 회전
import cv2

img = cv2.imread('./img.jpg')

rotate_180 = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전

cv2.imshow('img', img)
cv2.imshow('rotate_180', rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 시계 반대방향 90도 회전(시계 방향 270도 회전)
import cv2

img = cv2.imread('./img.jpg')

rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 시계 반대 방향으로 90도

cv2.imshow('img', img)
cv2.imshow('rotate_270', rotate_270)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 흑백으로 읽기
import cv2

img = cv2.imread('./img.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 불러온 이미지를 흑백으로 변경
import cv2

img = cv2.imread('./img.jpg')

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 이미지 변형(흐림) - 가우시안 블러
# 커널 사이즈 변화에 따른 흐림
import cv2

img = cv2.imread('./img.jpg')

# (3, 3), (5, 5), (7, 7)
kernel_3 = cv2.GaussianBlur(img, (3, 3), 0)
kernel_5 = cv2.GaussianBlur(img, (5, 5), 0)
kernel_7 = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imshow('img', img)
cv2.imshow('kernel_3', kernel_3)
cv2.imshow('kernel_5', kernel_5)
cv2.imshow('kernel_7', kernel_7)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 표준편차 변화에 따른 흐림
import cv2

img = cv2.imread('./img.jpg')

sigma_1 = cv2.GaussianBlur(img, (0, 0), 1) # sigmaX - 가우시안 커널의 x 방향의 표준 편차
sigma_2 = cv2.GaussianBlur(img, (0, 0), 2)
sigma_3 = cv2.GaussianBlur(img, (0, 0), 3)

cv2.imshow('img', img)
cv2.imshow('sigma_1', sigma_1)
cv2.imshow('sigma_2', sigma_2)
cv2.imshow('sigma_3', sigma_3)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 사다리꼴 이미지 펼치기
import cv2
import numpy as np

img = cv2.imread('./newspaper.jpg')

width, height = 640, 240 # 가로 크기 640, 세로 크기 240 으로 결과물 출력

src = np.array([[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype=np.float32) # Input 4개 지점
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32) # Output 4개 지점
# 좌상, 우상, 우하, 좌하 (시계 방향으로 4 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height)) # matrix 대로 변환을 함

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 사다리꼴 이미지 복원하기
import cv2
import numpy as np

img = cv2.imread('./newspaper.jpg')

width, height = 640, 240

src = np.array([[511, 352], [1008, 345], [1122, 584], [455, 594]], dtype=np.float32)
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(src, dst)   
warped = cv2.warpPerspective(img, matrix, (width, height))

# 복원을 위한 역행렬 (직사각형 → 사다리꼴) - img.shape[1]:가로 - img.shape[0]:세로
reverse_matrix = cv2.getPerspectiveTransform(dst, src)
recovered = cv2.warpPerspective(warped, reverse_matrix, (img.shape[1], img.shape[0]))  # 원본 크기로 복원

cv2.imshow('Original', img)
cv2.imshow('Warped (펼친 이미지)', warped)
cv2.imshow('Recovered (복원된 이미지)', recovered)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 회전된 이미지 올바로 세우기
import cv2
import numpy as np

img = cv2.imread('./poker.jpg')

width, height = 530, 710

src = np.array([[702, 143], [1133, 414], [726, 1007], [276, 700]], dtype=np.float32) # Input 4개 지점
dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32) # Output 4개 지점
# 좌상, 우상, 우하, 좌하 (시계 방향으로 4 지점 정의)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix 얻어옴
result = cv2.warpPerspective(img, matrix, (width, height)) # matrix 대로 변환을 함

cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Threshold(임계값)
import cv2

img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('binary', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Trackbar(0-255값 변화에 따른 변형 확인)
import cv2

def empty(pos):
    pass

img = cv2.imread('./book.jpg', cv2.IMREAD_GRAYSCALE)

name = 'Trackbar'
cv2.namedWindow(name)
cv2.createTrackbar('threshold', name, 127, 255, empty)
# bar 이름, 창 이름, 초기값, 최댓값, 이벤트 처리

while True:
    thresh = cv2.getTrackbarPos('threshold', name)   # bar 이름, 창의 이름
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    if not ret:
        break

    cv2.imshow(name, binary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()




# 그림판에서 제작한 이미지로 확인
import cv2

def empty(pos):
    # print(pos)
    pass

img = cv2.imread('threshold.png', cv2.IMREAD_GRAYSCALE)

name = 'Trackbar'
cv2.namedWindow(name)

cv2.createTrackbar('threshold', name, 127, 255, empty)   # bar 이름, 창의 이름, 초기값, 최대값, 이벤트 처리

while True:
    thresh = cv2.getTrackbarPos('threshold', name)   # bar 이름, 창의 이름
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    if not ret:
        break

    cv2.imshow('img', img)
    cv2.imshow(name, binary)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()




# 색 분류
import cv2

img = cv2.imread('./threshold.png', cv2.IMREAD_GRAYSCALE)

ret, binary1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)   # 진한 회색, 밝은 회색, 흰색
ret, binary2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)   # 밝은 회색, 흰색
ret, binary3 = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)   # 흰색

cv2.imshow('img', img)
cv2.imshow('binary1', binary1)
cv2.imshow('binary2', binary2)
cv2.imshow('binary3', binary3)

cv2.waitKey(0)
cv2.destroyAllWindows()




# Adaptive Threshold - adaptiveThreshold
import cv2

def empty(pos):
    # print(pos)
    pass

img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

name = 'Trackbar'
cv2.namedWindow(name)

# bar 이름, 창의 이름, 초기값, 최대값, 이벤트 처리
cv2.createTrackbar('block_size', name, 25, 100, empty)   # 홀수만 가능, 1보다는 큰 값
cv2.createTrackbar('c', name, 3, 10, empty)   # 일반적으로 양수의 값을 사용

while True:
    block_size = cv2.getTrackbarPos('block_size', name)   # bar 이름, 창의 이름
    c = cv2.getTrackbarPos('c', name)

    if block_size <= 1:   # 1 이하면 3 으로
        block_size = 3

    if block_size % 2 == 0:   # 짝수이면 홀수로
        block_size += 1

    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)

    cv2.imshow(name, binary)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()




# 오츠 알고리즘
import cv2

img = cv2.imread('./book.jpg', cv2.IMREAD_GRAYSCALE)

ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('otsu threshold ', ret)

cv2.imshow('img', img)
cv2.imshow('binary', binary)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()