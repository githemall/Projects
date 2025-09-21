# 나만의 이미지 분류 모델 만들기

# PyTorch 전이 학습(Transfer Learning)을 활용한 이미지 분류

**ResNet-50 모델을 기반으로 커스텀 데이터셋을 구축하고, 두 가지 다른 전이 학습 전략(Fine-tuning, Feature Extraction)의 성능과 효율성을 비교 분석한 프로젝트입니다.**

---

##  프로젝트 소개

이 프로젝트는 딥러닝 모델 개발의 전체 과정을 경험하고, 특히 전이 학습의 핵심 원리를 깊이 이해하는 것을 목표로 합니다. 직접 수집하고 정제한 **[동물 이미지]** 데이터셋을 사용하여, `ResNet-50` 모델을 기반으로 두 가지 다른 전략으로 모델을 학습시키고 그 결과를 심도 있게 분석했습니다.

* **학습 전략 1: 미세조정 (Fine-tuning)**
    * 미리 학습된 모델의 **모든 파라미터**를 나의 데이터셋에 맞게 조금씩 재학습합니다.
* **학습 전략 2: 특징 추출 (Feature Extraction)**
    * 모델의 **특징 추출기 부분은 그대로 얼려두고(freeze)**, 마지막 분류기만 새로 교체하여 학습합니다.

##  사용 기술 및 환경

* **언어:** `Python 3.9`
* **주요 라이브러리:**
    * `PyTorch` & `torchvision`
    * `NumPy`
    * `Matplotlib`
    * `Pillow (PIL)`
* **개발 환경:**
* **OS:** WSL2 (Ubuntu)
* **IDE:** Visual Studio Code
* **Language:** Python 3.9.22
* **GPU:** NVIDIA GeForce RTX 3060TI
* **Key Libraries:**
    * `torch==2.8.0+cu128`
    * `torchvision==0.23.0+cu128`
    * `numpy==1.24.3`
    * `matplotlib==3.9.4`
* **사전 학습 모델:** `ResNet-50`

---

##  데이터셋

* **주제:** **[동물 4종 (강아지, 고양이, 토끼, 햄스터) 분류]**
* **클래스:** **[`dog`, `cat`, `rabbit`, `hamster`]** (총 3개)
* **데이터 수:**
    * Training data: **[총 476장 (클래스당 약 120장)]**
    * Validation data: **[총 120장 (클래스당 30장)]**
* **데이터 수집:** **[Google 이미지 검색과 크롬 확장 프로그램을 사용하여 직접 수집]**
* **데이터 전처리:** `torchvision.transforms`를 사용하여 데이터 증강(Augmentation) 및 정규화(Normalization)를 적용했습니다.
    * **Train:** `RandomResizedCrop`, `RandomHorizontalFlip`
    * **Validation:** `Resize`, `CenterCrop`

---

##  주요 코드 설명

### 1. 모델 맞춤화 (Customizing)

`torchvision.models`에서 `ResNet-50` 모델을 불러온 뒤, 마지막 fully-connected layer를 우리의 클래스 개수에 맞게 교체했습니다.

```python
import torchvision.models as models

# 클래스 개수 가져오기
class_names = image_datasets['train'].classes
num_classes = len(class_names)

# 모델 불러오기
model = models.resnet50(weights='IMAGENET1K_V1')
# 마지막 레이어의 입력 피처 수 확인
num_ftrs = model.fc.in_features
# 새로운 분류기로 교체
model.fc = nn.Linear(num_ftrs, num_classes)
```

### 2. 특징 추출기 동결 (Freezing)
Feature Extraction 전략에서는 아래와 같이 모델의 모든 파라미터를 동결시키고, 옵티마이저에는 새로 추가된 model.fc.parameters()만 전달하여 마지막 레이어만 학습되도록 설정했습니다.
```python
# 모든 파라미터를 동결
for param in model.parameters():
    param.requires_grad = False
    
# 새로 추가한 fc layer는 기본적으로 requires_grad=True 상태

# fc layer의 파라미터만 학습하도록 옵티마이저 설정
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

---

##  학습 결과 및 분석 (ResNet-50)

`batch_size=32`, `num_epochs=25` 조건으로 두 가지 전략의 모델을 학습시킨 결과입니다.

| 학습 전략             | 최고 검증 정확도 (Best val Acc) | 학습 시간 (Training Time) | 특징                                                                                             |
| --------------------- | ------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------ |
| **Fine-tuning** | **97.5%** | **1분 43초** | 모델 전체를 데이터에 최적화하여 **가장 높은 성능**을 달성.                                         |
| **Feature Extraction**| 95.0%                           | 1분 26초                  | 마지막 레이어만 학습했음에도 **준수한 성능**을 보임.                                               |

**결론:** 예상대로 **Fine-tuning** 전략이 약간 더 높은 정확도를 보였습니다. 하지만 두 전략 간의 학습 시간 차이가 크지 않았는데, 이는 데이터 로딩 및 전처리 과정 대비 모델의 순수 계산 시간이 차지하는 비중이 크지 않음을 시사합니다. (자세한 내용은 아래 '배운 점 및 트러블슈팅' 참조)

---

##  예측 결과 예시

학습된 `model_ft` 모델이 검증용 이미지를 어떻게 예측하는지 시각화한 결과입니다.


<img width="158" height="151" alt="image" src="https://github.com/user-attachments/assets/5fb1ee7b-6eda-4005-a373-914b0b87cfbb" />
<img width="140" height="151" alt="image" src="https://github.com/user-attachments/assets/c5f2e582-c261-4938-9ebf-487e6d5c6130" />
<img width="135" height="151" alt="image" src="https://github.com/user-attachments/assets/04e28f53-4ffa-4369-8c71-095cec37a1ab" />
<img width="179" height="151" alt="image" src="https://github.com/user-attachments/assets/ff56e0c8-afbe-461b-be90-edab1f4fe2d3" />

---

##  배운 점 및 트러블슈팅

* **전이 학습 전략 비교:** 두 가지 핵심 전략(Fine-tuning, Feature Extraction)을 코드로 직접 구현하고, **정확도와 학습 시간의 트레이드오프 관계**를 실험을 통해 명확히 이해했습니다.

* **성능 병목(Bottleneck) 분석:**
    * **현상:** 초기 `ResNet-18` 모델 실험에서 두 전략의 학습 시간 차이가 거의 없는 현상을 발견했습니다. 배치 사이즈를 늘리고, 더 무거운 `ResNet-50` 모델로 변경하는 추가 실험을 진행했습니다.
    * **원인:** 모델의 순수 계산 시간보다 **데이터 로딩 및 전처리 과정이 성능 병목**으로 작용하고 있음을 확인했습니다. 제 개발 환경(GPU)에서는 ResNet-50 정도의 연산도 매우 빠르게 처리되어, 데이터 I/O 시간이 전체 학습 시간을 좌우하는 주요인이었습니다.
    * **교훈:** 모델의 성능 튜닝 시, 코드뿐만 아니라 하드웨어와 데이터 파이프라인까지 종합적으로 고려해야 한다는 실무적인 교훈을 얻었습니다.

* **이미지 채널(Channel) 에러 해결:**
    * **문제:** 단일 이미지 예측 시, 4채널(RGBA) PNG 파일이 3채널(RGB)을 가정하는 `Normalize` 단계에서 `RuntimeError`를 발생시켰습니다.
    * **해결:** `Image.open()` 직후 **`.convert('RGB')`**를 추가하여 모든 입력 이미지의 채널을 3개로 통일시켜 문제를 해결했습니다.

---

##  관련 링크

* **블로그 포스트:** [Velog - 내 GPU는 너무 빨랐다: PyTorch 전이 학습 성능 비교 삽질기](https://velog.io/@githemall/%EB%82%B4-GPU%EB%8A%94-%EB%84%88%EB%AC%B4-%EB%B9%A8%EB%9E%90%EB%8B%A4-PyTorch-%EC%A0%84%EC%9D%B4-%ED%95%99%EC%8A%B5-%EC%84%B1%EB%8A%A5-%EB%B9%84%EA%B5%90-%EC%82%BD%EC%A7%88%EA%B8%B0)
* **참고 튜토리얼:** [PyTorch - Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
