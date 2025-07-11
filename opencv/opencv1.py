# 이미지 출력
import cv2

img = cv2.imread('./img.jpg')   # 해당 경로의 파일 읽어오기
cv2.imshow('img', img)   # img 라는 이름의 창에 img 를 표시
cv2.waitKey(0)   # 지정된 시간(ms) 동안 사용자 키 입력 대기
cv2.destroyAllWindows()   # 모든 창 닫기




# 읽기 옵션
import cv2
img_color = cv2.imread('./img.jpg', cv2.IMREAD_COLOR)   # RGB
img_gray = cv2.imread('./img.jpg', cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('./img.jpg', cv2.IMREAD_UNCHANGED)   # 알파채널 포함

cv2.imshow('img_color', img_color)
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_unchanged', img_unchanged)
cv2.waitKey(0)
cv2.destroyAllWindows()




# shape
import cv2
img = cv2.imread('./img.jpg')
img.shape   # 세로, 가로, Channel




# 동영상 파일 출력
import cv2

cap = cv2.VideoCapture('./video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('더 이상 가져올 프레임이 없어요')
        break

    cv2.imshow('frame', frame)   # 일반 환경에서 화면에 출력하는 방법

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('사용자 입력에 의해 종료합니다')
        break

cap.release()
cv2.destroyAllWindows()




# (Jupyter Notebook_only) 동영상 파일 출력
from IPython.display import HTML
from base64 import b64encode

# 1) 재생할 동영상 파일 경로 지정
video_path = 'video.mp4'  # Colab 작업 디렉토리에 업로드된 파일명

# 2) 파일을 바이너리로 읽어 Base64로 인코딩
with open(video_path, 'rb') as f:
    video_bytes = f.read()
data_url = "data:video/mp4;base64," + b64encode(video_bytes).decode()

# 3) HTML5 <video> 태그로 삽입
HTML(f"""
<video width="640" height="360" controls autoplay loop>
  <source src="{data_url}" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
""")




# 카메라 출력
import cv2
cap = cv2.VideoCapture(0)   # 0번째 카메라 장치 (Device ID)

if not cap.isOpened():   # 캠이 열리지 않은 경우
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera',frame)
    if cv2.waitKey(1) == ord('q'):   # q를 입력하면 종료
        break

cap.release()
cv2.destroyAllWindows()




# 카메라 프레임 저장
import cv2
import os
import time

def auto_capture_frames(interval=1.0, max_frames=50):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다!")
        return
    
    if not os.path.exists('auto_frames'):
        os.makedirs('auto_frames')
    
    frame_count = 0
    last_save_time = time.time()
    
    print(f"{interval}초마다 자동 저장, 최대 {max_frames}개. 'q'로 종료")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Auto Capture', frame)
        
        current_time = time.time()   # 일정 간격마다 자동 저장
        if current_time - last_save_time >= interval:
            filename = f'auto_frames/frame_{frame_count:04d}.jpg'
            cv2.imwrite(filename, frame)
            print(f"자동 저장: {filename}")
            frame_count += 1
            last_save_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"완료! {frame_count}개 프레임 저장됨")

auto_capture_frames(interval=1.0, max_frames=30)   # 1초마다 자동 저장




# 저장된 프레임으로 영상 출력
import cv2
import os

def create_video_from_images(image_folder, output_video, fps=10):
    """
    저장된 이미지를 동영상으로 변환하는 함수.
    Args:
        - image_folder: 이미지가 저장된 폴더 경로
        - output_video: 생성할 동영상 파일 이름
        - fps: 초당 프레임 수 (기본값 10)
    """
    # 폴더 내 이미지 파일 리스트 가져오기
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()   # 이미지 파일 이름 순서대로 정렬
    
    if not images:
        print("No images found in the folder.")
        return
    
    # 첫 번째 이미지로 프레임 크기 결정
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height)
    
    print(f"비디오 해상도: {width}x{height}")
    
    # 동영상 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # .mp4 파일 포맷
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    
    # 각 이미지를 동영상에 추가
    for i, image in enumerate(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)   # 동영상에 프레임 추가
    
    video.release()
    print(f"동영상 저장 완료: {output_video}")

# 사용 예시
image_folder = "./auto_frames/"  # 저장한 프레임 폴더 경로
output_video = image_folder + "output_video.mp4"    # 생성할 동영상 파일 이름
fps = 10                             # 초당 프레임 수

create_video_from_images(image_folder, output_video, fps)




# 생성된 동영상 파일 출력
import cv2

cap = cv2.VideoCapture('auto_frames/output_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('더 이상 가져올 프레임이 없어요')
        break

    cv2.imshow('frame', frame)   # 일반 환경에서 화면에 출력하는 방법

    key = cv2.waitKey(120) & 0xFF  # 120ms 대기

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('사용자 입력에 의해 종료합니다')
        break

cap.release()
cv2.destroyAllWindows()