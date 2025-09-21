# AI 기반 한국어 영화 리뷰 감성 분석 API

**Hugging Face Transformers의 사전 학습 모델(Ko-ELECTRA)을 Naver Movie Review Corpus(NSMC) 데이터셋으로 파인튜닝하고, 이를 FastAPI를 통해 실시간 예측 API로 배포한 프로젝트입니다.**

[Image of a web browser showing the FastAPI docs page]

---

## 프로젝트 목표

1.  **모델링:** 단순한 머신러닝(TF-IDF) 베이스라인 모델과 최신 딥러닝(BERT 계열) 모델의 성능을 비교 분석하여, 자연어 처리(NLP) 모델의 발전 과정을 이해합니다.
2.  **API 개발:** 완성된 AI 모델을 실제 서비스에서 사용할 수 있도록, FastAPI를 활용하여 RESTful API 서버를 구축합니다.
3.  **문제 해결:** 모델 학습 및 서버 배포 과정에서 발생하는 다양한 문제(환경 설정, 성능 병목, 네트워크 등)를 진단하고 해결하는 능력을 기릅니다.

---

## 사용 기술 및 환경

* **OS:** **[WSL2 Ubuntu]**
* **Language:** Python 3.x
* **AI/ML:** `PyTorch`, `Hugging Face Transformers`, `Scikit-learn`, `KoNLPy`
* **API Server:** `FastAPI`, `Uvicorn`
* **Libraries:** `Pandas`, `NumPy`, `requests` (자세한 내용은 `requirements.txt` 참조)
* **사전 학습 모델:** `beomi/kcELECTRA-base-v2022`

---

## 🗂️ 데이터셋

* **데이터:** [Naver Sentiment Movie Corpus (NSMC)](https://github.com/e9t/nsmc)
* **구성:** Training data 150K개, Test data 50K개
* **데이터 정제:**
    * `Pandas`를 활용하여 결측치(NaN) 및 중복 리뷰를 제거하여 약 14.6만 개의 훈련용 데이터를 사용했습니다.
    * 전처리 속도 향상을 위해, TF-IDF 변환 결과 및 정제된 데이터셋을 `joblib`, `scipy.sparse`, `csv` 파일로 저장하여 재사용하는 파이프라인을 구축했습니다.

---

## 📊 모델 성능 비교

동일한 데이터셋으로 두 가지 다른 모델을 학습시켜 성능을 비교했습니다.

| 모델 종류                         | 주요 기술                                  | 최종 테스트 정확도 (Accuracy) | 특징                                         |
| --------------------------------- | ------------------------------------------ | ------------------------------- | -------------------------------------------- |
| **Baseline Model** | KoNLPy(Okt), Scikit-learn(TF-IDF, Logistic Regression) | **86.75%** | 빠르고 가볍지만, 문맥 이해에 한계가 있음.    |
| **Fine-tuned Model (🔥 최종 모델)** | **Hugging Face Transformers (Ko-ELECTRA)** | **9
