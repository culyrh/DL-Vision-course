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

img = cv2.imread('./img.jpg', cv2.IMREAD_GRAYSCALE)   # 흑백으로 이미지 불러오기
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
fps    = cap.get(cv2.CAP_PROP_FPS) * 2   # 재생 속도 2배

# 4) 출력 파일 설정 - 객체 생성
output_path = './video_fast.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 5) 프레임 처리 루프 - 덮어씌움
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)        # 영상 프레임만 저장 (소리 제외)
    # cv2.imshow('frame', frame)       # 화면 출력 (옵션)


# 6) 리소스 해제
cap.release()
out.release()


# 7) 생성된 AVI 파일 다운로드
print(output_path)
# files.download(output_path)   # (코랩) 출력

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