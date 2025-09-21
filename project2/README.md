# AI 기반 한국어 영화 리뷰 감성 분석 API

**Hugging Face Transformers의 사전 학습 모델(Ko-ELECTRA)을 Naver Movie Review Corpus(NSMC) 데이터셋으로 파인튜닝하고, 이를 FastAPI를 통해 실시간 예측 API로 배포한 프로젝트입니다.**



---

## 프로젝트 목표

1.  **모델링:** 단순한 머신러닝(TF-IDF) 베이스라인 모델과 최신 딥러닝(BERT 계열) 모델의 성능을 비교 분석하여, 자연어 처리(NLP) 모델의 발전 과정을 이해합니다.
2.  **API 개발:** 완성된 AI 모델을 실제 서비스에서 사용할 수 있도록, FastAPI를 활용하여 RESTful API 서버를 구축합니다.
3.  **문제 해결:** 모델 학습 및 서버 배포 과정에서 발생하는 다양한 문제(환경 설정, 프로세스 행(hang), 네트워크 등)를 진단하고 해결하는 능력을 기릅니다.

---

## 사용 기술 및 환경

* **OS:** **[WSL2 Ubuntu]**
* **Language:** Python 3.10.12
* **AI/ML:** `PyTorch`, `Hugging Face Transformers`, `Scikit-learn`, `KoNLPy`
* **API Server:** `FastAPI`, `Uvicorn`
* **Libraries:** `Pandas`, `NumPy`, `requests`
* **사전 학습 모델:** `beomi/kcELECTRA-base-v2022`

---

## 데이터셋

* **데이터:** [Naver Sentiment Movie Corpus (NSMC)](https://github.com/e9t/nsmc)
* **구성:** Training data 150K개, Test data 50K개
* **데이터 정제:**
    * `Pandas`를 활용하여 결측치(NaN) 및 중복 리뷰를 제거하여 약 14.6만 개의 훈련용 데이터를 사용했습니다.
    * 전처리 속도 향상을 위해, TF-IDF 변환 결과 및 정제된 데이터셋을 `joblib`, `scipy.sparse`, `csv` 파일로 저장하여 재사용하는 파이프라인을 구축했습니다.

---

## 모델 성능 비교

동일한 데이터셋으로 두 가지 다른 모델을 학습시켜 성능을 비교했습니다.

| 모델 종류                         | 주요 기술                                  | 최종 테스트 정확도 (Accuracy) | 특징                                         |
| --------------------------------- | ------------------------------------------ | ------------------------------- | -------------------------------------------- |
| **Baseline Model** | KoNLPy(Okt), Scikit-learn(TF-IDF, Logistic Regression) | **86.75%** | 빠르고 가볍지만, 문맥 이해에 한계가 있음.    |
| **Fine-tuned Model (최종 모델)** | **Hugging Face Transformers (Ko-ELECTRA)** | **92.57%** | 문맥을 깊이 이해하여 월등히 높은 성능을 보임. |

---

## 설치 및 실행 방법


  **API 서버 실행**
    - * (WSL 환경) Tokenizer 병렬 처리 문제 해결을 위해 아래 명령어를 먼저 실행합니다.
        ```bash
        export TOKENIZERS_PARALLELISM=false
        ```
    - * Uvicorn으로 서버를 실행합니다.
        ```bash
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```

  **API 테스트**
    * 웹 브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하여 API를 테스트할 수 있습니다.

---

## API 엔드포인트

### `/predict`

* **Method:** `POST`
* **Request Body:**
    ```json
    {
      "text": "여기에 영화 리뷰를 입력하세요."
    }
    ```
* **Success Response (200):**
    ```json
    {
      "prediction": "긍정 😄",
      "probability": 0.9257
    }
    ```
* **`curl` 예시:**
    ```bash
    curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
    -H "Content-Type: application/json" \
    -d '{"text": "배우들 연기가 정말 미쳤네요... 몰입감 최고!"}'
    ```

---

## 배운 점 및 트러블슈팅

이번 프로젝트를 통해 모델링 외에도 다양한 실무적 경험을 쌓을 수 있었습니다.

1.  **환경 설정:** `KoNLPy` 사용을 위한 **Java(JDK) 및 `JAVA_HOME` 환경 변수 설정** 방법을 익혔습니다. 이를 통해 Python 생태계가 다른 언어(Java)와 어떻게 상호작용하는지 이해하는 좋은 계기가 되었습니다.

2.  **프로세스 행(Hang) 디버깅:**
    * **학습 평가:** `DataLoader`의 `num_workers > 0` 설정이 Jupyter/WSL 환경에서 프로세스 교착 상태(deadlock)를 일으켜 평가가 멈추는 현상을 발견하고, `num_workers=0`으로 설정하여 해결했습니다.
    * **API 서버:** FastAPI에서 `Tokenizer`의 병렬 처리 기능이 서버 프로세스와 충돌하여 예측 요청 시 서버가 멈추는 문제를 `export TOKENIZERS_PARALLELISM=false` 환경 변수 설정으로 해결했습니다.

3.  **네트워크 디버깅:** WSL2 환경에서 Windows 브라우저로의 접속이 간헐적으로 실패하는 문제를 해결하기 위해, **`curl` 테스트, `wsl --shutdown`, 방화벽 규칙 추가, 서드파티 보안 프로그램(AdGuard) 필터링 예외 처리** 등 깊이 있는 네트워크 디버깅을 경험했습니다. 이 과정을 통해 문제의 원인을 체계적으로 좁혀나가는 능력을 길렀습니다.

4.  **라이브러리 업데이트 대응:** `transformers` 라이브러리 버전업으로 인해 `AdamW`의 위치가 변경되고(`torch.optim`), 인자가 달라지는(`correct_bias` 제거) 문제에 대응하며, 라이브러리 공식 문서를 참조하는 것의 중요성을 깨달았습니다.

---

## 🔗 관련 링크

* **블로그 포스트:** **[수정: Velog - BERT 모델 API 배포 및 삽질기](<블로그 링크>)**
