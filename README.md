# Pokemon Classifier

> ResNet18 transfer learning으로 150종 포켓몬을 분류하는 이미지 분류기.

---

## 1. Demo
<img width="1905" height="855" alt="Image" src="https://github.com/user-attachments/assets/089b561a-d6bf-4b8c-ad3d-5228ba586a95" />


---

## 2. 프로젝트 개요

**실험 목표:** Pretrained backbone과 fine-tuning를 변수로 하여 이 범위가 포켓몬 이미지 분류 성능에 어떤 영향을 주는지 비교.

ImageNet으로 학습된 ResNet18을 기반으로, 4가지 실험을 수행합니다:

- (1) **Scratch** — pretrained 가중치 사용하지 않고 처음부터 학습
- (2) **Linear probe** — pretrained backbone 동결, FC만 학습
- (3) **Partial fine-tuning** — pretrained backbone 중 마지막 block(`layer4`) + FC 학습
- (4) **Full fine-tuning** — pretrained 가중치를 시작점으로 모든 layer 학습

**학습 결과:** Best 모델은 **Exp 4: ResNet18-full-FT**
-> test accuracy **94.43%** 달성.


---

## 3. 데이터셋

<img width="1907" height="800" alt="Image" src="https://github.com/user-attachments/assets/b27f240e-6dee-4d72-8c94-131c4e12e307" />

- **출처:** [7,000 Labeled Pokemon (Kaggle)](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- **클래스 수:** 150
- **전체 이미지 수:** 약 7,000장

---

## 4. 실험 설정

### 4-1. 4개 실험 비교

| Exp | Backbone | Pretrained | Fine-tuning 범위 | Trainable params |
|:---:|:---:|:---:|:---:|:---:|
| 1 | ResNet18 | X (scratch) | 전체 | 100% |
| 2 | ResNet18 | O | Linear probe (FC만) | 약 0.7% |
| 3 | ResNet18 | O | Partial (`layer4` + FC) | 약 75.3% |
| 4 | ResNet18 | O | Full fine-tuning | 100% |

### 4-2. 공통 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (weight decay 1e-4) |
| LR scheduler | CosineAnnealingLR |
| Learning rate | scratch / linear probe = 1e-3, partial / full = 1e-4 |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Batch size | 32 |
| Epochs | scratch 30, 그 외 15 (early stopping patience 5) |
| Image size | 224 × 224 |
| Seed | 42 |

---

## 5. 결과

### 5-1. 성능 비교 (Test set)

<img width="812" height="180" alt="Image" src="https://github.com/user-attachments/assets/0a0bbcc1-e859-4b9e-b748-367163f23edb" />

> Precision / Recall / F1은 모두 **macro average** 기준.

### 5-2. Learning curves

<img width="1389" height="490" alt="Image" src="https://github.com/user-attachments/assets/a5efe43d-232d-4b5a-96f1-aab6da61f681" />

- 실선: validation, 점선: train
- 색상별로 4개 실험 구분

### 5-3. 분석

- **Pretraining 효과 (Exp 1 vs Exp 4):** 같은 ResNet18 구조에서도 ImageNet pretrained 가중치 사용 여부에 따라 test accuracy가 **75.86% → 94.43%**, 약 **18.6%p** 차이가 발생했습니다. 클래스당 ~47장에 불과한 작은 데이터셋에서는 pretraining 사용 여부가 거의 결정적인 영향을 미쳤으며, scratch 학습은 30 epoch을 돌려도 train accuracy 99.9% / val accuracy 77% 수준의 심각한 overfitting을 보입니다.
- **Fine-tuning 범위 효과 (Exp 2 → 3 → 4):** 학습 범위를 넓힐수록 정확도가 계단처럼 증가했습니다 (Linear probe **82.11%** → Partial **92.28%** → Full **94.43%**). 가장 큰 향상은 Linear probe → Partial 구간 (**+10.2%p**)에서 발생했고, Partial → Full 구간은 상대적으로 작은 차이 (**+2.2%p**)에 그쳤다. 즉, 마지막 block(`layer4`)만 fine-tuning해도 full fine-tuning에 가까운 성능을 얻을 수 있었습니다.
- **흥미로운 결과**
  - **Linear probe만으로도 82%** 도달: backbone을 전혀 학습시키지 않고도 ImageNet feature가 포켓몬 도메인에 강하게 전이되었음을 보여준다.
  - **Full fine-tuning은 4~5 epoch 만에 거의 수렴** (val acc 94%에 도달 후 plateau): 작은 데이터셋이라 추가 epoch이 큰 의미가 없으며, partial/full 모두 early stopping이 작동했습니다.
  - Partial과 Full의 학습 곡선/최종 성능이 매우 유사한 점은, **계산 비용을 줄이고 싶을 때 partial fine-tuning이 합리적인 선택**이 될 수 있음을 보여준다고 생각됩니다.

### 5-4. 오분류 분석 (best 모델 기준)

자주 헷갈리는 클래스 top-N과 recall이 가장 낮은 클래스 분석은 노트북의 6-3 셀에서 확인 가능.

대표적 오분류 예시:

| 횟수 | True | Predicted |
|:---:|:---|:---|
| 3 | Rapidash | Moltres |
| 3 | Alakazam | Kadabra |
| 2 | Ponyta | Rapidash |


---

## 6. 사용 방법

### 6-1. 학습 (Google Colab)

1. [pokemon_classification.ipynb](pokemon_classification.ipynb)를 [Google Colab](https://colab.research.google.com/)에 업로드
2. **런타임 → 런타임 유형 변경 → T4 GPU** 선택
3. 셀을 위에서 아래로 순서대로 실행 (전체 약 50분 소요)
4. 마지막 셀에서 생성되는 `best_model.pt`를 다운로드해 저장

### 6-2. 데모 실행 (로컬)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. best_model.pt 를 app.py 와 같은 폴더에 둔 뒤
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501`이 열리고, 이미지를 업로드하면 Top-5 포켓몬 예측 결과가 표시됩니다.

---

## 7. 환경 / 의존성

- Python 3.10+
- 주요 라이브러리는 torch, torchvision, kagglehub, scikit-learn 등이며 전체 목록은 requirements.txt 참고.
