import os
from ultralytics import YOLO

# --- 1. 설정 (Configuration) ---
# (사용자가 직접 수정해야 하는 부분)

# 1. 훈련된 모델 파일(.pt)의 경로
# (예: 'YOLO_Results/recycle_detection_v11/weights/best.pt')
MODEL_PATH = '/home/asd/projects/Portfolio/graduation/Recycle_detection_11s/runs/detect/train6/weights/best.pt'
#MODEL_PATH = '/home/asd/projects/Portfolio/graduation/Recycle_detection_11n/best.pt'
# 2. 검증(테스트)할 이미지 파일의 경로
# (예: 'test_images/my_bottle.jpg')
TEST_IMAGE_PATH = '/home/asd/projects/Portfolio/graduation/test_images/test.jpg'
# (미리 'test_images' 폴더를 만들고 이미지를 넣어두세요)

# 3. 신뢰도 임계값 (Confidence Threshold)
# 0.4 = 40% 이상 확신하는 것만 표시 (숫자를 낮추면 더 많은 객체를 찾지만, 실수가 잦아짐)
CONF_THRESHOLD = 0.4

# -----------------------------------

print(f"Loading model from: {MODEL_PATH}")
print(f"Running prediction on: {TEST_IMAGE_PATH}")

try:
    # 2. 모델 로드
    model = YOLO(MODEL_PATH)

    # 3. 예측 실행
    # 'save=True' : 'runs/detect/predict/' 폴더에 결과 이미지를 자동 저장
    # 'conf=...' : 설정한 신뢰도 임계값 적용
    results = model.predict(
        source=TEST_IMAGE_PATH,
        save=True,
        conf=CONF_THRESHOLD
    )

    print("\nPrediction complete!")
    print(f"Results saved to the 'runs/detect/predict/' directory.")

    # 4. (선택 사항) 결과에 대한 상세 정보 출력
    for r in results:
        print(f"\nImage: {r.path}")
        print(f"Found {len(r.boxes)} objects:")
        for box in r.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            print(f"  - {class_name} (Confidence: {confidence:.2f})")

except FileNotFoundError as e:
    print(f"\n[Error] File not found: {e.filename}")
    print("MODEL_PATH와 TEST_IMAGE_PATH 경로가 올바른지 다시 확인해주세요.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
