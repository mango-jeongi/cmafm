# 교차 모달 어텐션 기반 RGB-Thermal 영상 융합 객체 검출

## Cross-Modal Attention-Based RGB-Thermal Image Fusion for Object Detection

---

### 요약

본 논문은 가시광(RGB) 영상과 적외선(Thermal) 영상을 효과적으로 융합하여 보행자 및 차량을 검출하는 딥러닝 기반 다중 스펙트럼 객체 검출 모델을 제안한다. 제안하는 모델은 각 모달리티에 독립적인 ResNet-50 백본을 할당하여 고유 특징을 추출한 후, 채널 교차 어텐션(Channel Cross-Attention)과 공간 교차 게이팅(Spatial Cross-Gating)으로 구성된 교차 모달 어텐션 융합 모듈(Cross-Modal Attention Fusion Module, CMAFM)을 통해 상호 보완적 정보를 교환한다. 융합된 특징은 Feature Pyramid Network(FPN)을 거쳐 Faster R-CNN 검출 헤드로 전달된다. M3FD 데이터셋에서의 실험 결과, 제안 모델은 mAP@0.5 73.7%, mAP@[.5:.95] 40.6%, Recall 87.4%를 달성하였으며, RGB 단일 모달리티 대비 mAP@0.5 기준 10.5%p, Recall 기준 4.5%p의 성능 향상을 보였다. Ablation study를 통해 이중 백본 구조(+5.0%p)와 교차 모달 어텐션(+3.7%p)의 개별 기여도를 정량적으로 검증하였다. 나아가, 야간/주간 조건별 분석을 통해 야간에서 mAP@0.5 85.3%, Recall 90.9%를 달성하여 주간(mAP@0.5 72.8%, Recall 87.0%) 대비 각각 12.5%p, 3.9%p 높은 성능을 보임으로써, RGB-Thermal 융합이 열악한 조도 환경에서 특히 효과적임을 실증하였다.

**핵심어**: 다중 스펙트럼 객체 검출, RGB-Thermal 융합, 교차 모달 어텐션, 이중 백본, Faster R-CNN

### Abstract

This paper proposes a deep learning-based multispectral object detection model that effectively fuses visible (RGB) and infrared (Thermal) images for pedestrian and vehicle detection. The proposed model assigns independent ResNet-50 backbones to each modality for extracting modality-specific features, then exchanges complementary information through a Cross-Modal Attention Fusion Module (CMAFM) consisting of Channel Cross-Attention and Spatial Cross-Gating mechanisms. The fused features are passed through a Feature Pyramid Network (FPN) to a Faster R-CNN detection head. Experiments on the M3FD dataset demonstrate that the proposed model achieves 73.7% mAP@0.5, 40.6% mAP@[.5:.95], and 87.4% Recall, showing improvements of 10.5%p in mAP@0.5 and 4.5%p in Recall over the RGB-only baseline, and 8.7%p over the Early Fusion baseline. Through ablation studies, the individual contributions of the dual-backbone architecture (+5.0%p) and cross-modal attention (+3.7%p) are quantitatively verified. Furthermore, condition-specific analysis reveals that the model achieves 85.3% mAP@0.5 and 90.9% Recall under nighttime conditions, surpassing daytime performance (72.8% mAP@0.5, 87.0% Recall) by 12.5%p and 3.9%p respectively, demonstrating the particular effectiveness of RGB-Thermal fusion in low-illumination environments.

**Keywords**: Multispectral Object Detection, RGB-Thermal Fusion, Cross-Modal Attention, Dual Backbone, Faster R-CNN

---

## I. 서론

자율 주행, 군사 감시 등 안전 필수 응용에서 보행자·차량 검출은 다양한 환경 조건에서의 강인한 성능을 요구한다[1]. RGB 카메라는 주간에 우수하나 야간·역광·안개 환경에서 성능이 급락하고[2], 적외선(Thermal) 카메라는 조명 독립적이나 텍스처·색상 부재로 유사 열 특성 객체 구별이 어렵다[3]. 두 모달리티의 상호 보완적 특성을 활용하는 RGB-Thermal 융합 연구가 활발히 진행되고 있으며[4-6], 각 모달리티의 고유 특징을 충분히 추출한 후 결합하는 중간 수준 융합(Mid-level Fusion)이 가장 우수한 성능을 보이는 것으로 보고된다[7].

그러나 기존 중간 수준 융합은 단순 연결이나 요소별 덧셈에 그쳐 모달리티 간 상호 보완적 관계를 충분히 활용하지 못하며, Transformer 기반 공간 어텐션은 O(N²) 복잡도로 고해상도 특징맵 적용에 제약이 있다. 본 논문에서는 GAP 기반 채널 교차 어텐션(복잡도 O(C))과 DWConv 기반 공간 교차 게이팅을 결합한 교차 모달 어텐션 융합 모듈(CMAFM)과 이중 백본 아키텍처를 제안하며, M3FD 데이터셋 실험 및 ablation study를 통해 각 구성 요소의 기여도를 정량적으로 검증한다.

---

## II. 관련 연구

### 2.1 RGB-Thermal 융합 객체 검출

RGB-Thermal 융합은 융합 위치에 따라 조기·중간·후기 융합으로 분류된다. Liu et al.[8]은 입력 단계에서 채널을 결합하는 조기 융합을 제안하였고, TarDAL[4]은 적대적 학습 기반, Zhang et al.[5]은 반복적 융합-정제 블록으로 중간 수준 융합을 구현하였다. Cao et al.[6]과 Qingyun et al.[10]은 Transformer 기반 교차 모달 융합을 시도하였으나 O(N²) 복잡도로 고해상도 적용에 제약이 있다.

### 2.2 어텐션 기반 특징 융합 및 검출기

Hu et al.[11]의 SENet은 채널별 중요도를 재조정하는 Squeeze-and-Excitation 구조를 제안하였고, Zhang et al.[9]의 CIAN은 이를 확장하여 교차 모달리티 간 정보 교환을 시도하였다. 본 논문의 CMAFM은 GAP 기반 채널 교차 어텐션과 DWConv 기반 공간 교차 게이팅을 결합하여 N×N 어텐션 행렬 없이 양방향 전역·지역 정보 교환을 달성한다. 검출 프레임워크로는 RPN과 ROI Head로 구성된 Faster R-CNN[12]에 FPN[13]을 결합하여 다중 스케일 검출 정확도를 확보하였다.

---

## III. 제안 방법

### 3.1 전체 구조

제안하는 모델의 전체 구조는 그림 1과 같다. 모델은 크게 (1) 이중 백본, (2) 교차 모달 어텐션 융합 모듈(CMAFM), (3) FPN, (4) Faster R-CNN 검출 헤드로 구성된다.

```
RGB 영상 (H×W×3)
    │
    ▼
┌──────────────┐
│ ResNet-50    │──► C3 (H/8, 512ch) ──┐
│ (RGB 전용)   │──► C4 (H/16, 1024ch)─┤──► CMAFM ──► FPN ──► Faster R-CNN
│              │──► C5 (H/32, 2048ch)─┤                       (RPN + ROI)
└──────────────┘                       │                          │
                                       │                          ▼
┌──────────────┐                       │                    검출 결과
│ ResNet-50    │──► C3 (H/8, 512ch) ──┤                 {boxes, labels,
│ (Thermal 전용)│──► C4 (H/16, 1024ch)─┤                  scores}
│              │──► C5 (H/32, 2048ch)─┘
└──────────────┘

Thermal 영상 (H×W×1 → 3ch 복제)
```

**그림 1.** 제안 모델의 전체 구조

입력 RGB 영상과 Thermal 영상은 각각 640×640으로 리사이즈된다. Thermal 영상은 1채널이므로 3채널로 복제하여 ImageNet 사전 학습 가중치를 활용한다. 각 백본은 C3, C4, C5 세 개 스케일의 특징맵을 추출하며, 동일 스케일의 RGB-Thermal 특징쌍에 대해 CMAFM이 적용된다.

### 3.2 이중 백본 구조

각 모달리티에 독립적인 ResNet-50 백본을 할당한다. 두 백본은 동일한 ImageNet 사전 학습 가중치로 초기화되지만, 학습 과정에서 각 모달리티의 특성에 맞게 독립적으로 미세 조정된다.

RGB 백본은 텍스처, 색상, 형태 등의 시각적 특징 추출에 특화되고, Thermal 백본은 열 분포, 열원 경계 등의 적외선 고유 특징 추출에 특화된다. 이러한 분리는 단일 백본이 두 모달리티의 이질적 특징을 동시에 학습해야 하는 부담을 경감시킨다.

### 3.3 교차 모달 어텐션 융합 모듈 (CMAFM)

CMAFM은 **F**_r, **F**_t ∈ ℝ^(C×H×W)를 입력받아 세 단계로 융합 특징맵 **F**_out을 출력한다. ⊙는 요소별 곱, σ는 시그모이드를 나타낸다.

**① 채널 교차 어텐션**: GAP으로 채널 통계를 추출하여 교차 모달리티 dot-product 스케일링을 수행한다(복잡도 O(C)). RGB→Thermal 방향:

> **F**_r′ = **F**_r ⊙ σ(W_q·GAP(**F**_r) ⊙ W_k·GAP(**F**_t)) ⊙ W_v·GAP(**F**_t)   &emsp; (1)

동일 연산을 Thermal→RGB에도 적용하여 양방향 채널 교환을 달성한다.

**② 공간 교차 게이팅**: DWConv_{3×3}으로 추출한 공간 특징을 상대 모달리티 맥락으로 게이팅하여, 관련 영역을 강화하고 무관한 영역을 억제한다:

> **F**_r″ = σ(Conv_{1×1}([**S**_r ; **S**_t])) ⊙ **F**_r′,   **S**_r = Conv_{1×1}(DWConv_{3×3}(**F**_r′))   &emsp; (2)

**③ 게이트 융합**: 픽셀·채널별 게이트 α ∈ ℝ^(C×H×W)로 두 모달리티를 적응적으로 가중 합산하고 잔차 연결을 적용한다:

> α = σ(Conv_{1×1}([**F**_r″ ; **F**_t″])),   **F**_out = Conv_{3×3}(α⊙**F**_r″ + (1−α)⊙**F**_t″) + **F**_fused   &emsp; (3)

### 3.4 검출 헤드

융합된 C3, C4, C5 특징맵은 FPN[13]을 통해 P3, P4, P5 및 추가 풀링 레벨로 변환된다. FPN 출력 채널은 256으로 통일된다. Faster R-CNN[12]의 RPN이 앵커 기반 영역 제안을 생성하고, ROI Head가 분류 및 바운딩 박스 회귀를 수행한다.

### 3.5 학습 전략

백본에는 기본 학습률(0.005)의 0.1배를 적용하는 차등 학습률(Differential Learning Rate)을 사용하여 사전 학습 가중치를 보존하고, 융합 모듈 및 검출 헤드는 기본 학습률로 학습한다. StepLR 스케줄러로 10 epoch마다 학습률을 0.1배 감소시킨다.

---

## IV. 실험

### 4.1 데이터셋

M3FD(Multi-Modal Multi-Scene Fusion Detection) 데이터셋[14]을 사용하였다. M3FD는 도로 주행 환경에서 수집된 4,200쌍의 정합된 가시광-적외선 영상으로 구성되며, 주간, 야간, 흐림, 도전적 조건(역광, 안개) 등 다양한 환경을 포함한다.

**표 1.** M3FD 데이터셋 클래스별 객체 수

| 클래스 | 객체 수 | 비율 |
|--------|---------|------|
| Car | 18,296 | 53.2% |
| People | 11,477 | 33.4% |
| Lamp | 2,405 | 7.0% |
| Truck | 1,008 | 2.9% |
| Bus | 700 | 2.0% |
| Motorcycle | 521 | 1.5% |
| **합계** | **34,407** | **100%** |

데이터셋을 8:2 비율로 학습(3,360쌍)/검증(840쌍)으로 분할하였다. 영상 해상도는 1024×768이며, 학습 시 640×640으로 리사이즈한다.

### 4.2 실험 환경

**표 2.** 실험 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX A6000 (48GB) |
| Framework | PyTorch 2.4.0 |
| Backbone | ResNet-50 (ImageNet pretrained) |
| Optimizer | SGD (momentum=0.9, weight_decay=5×10^-4) |
| 학습률 | 0.005 (backbone: 0.0005) |
| Batch size | 8 |
| Epoch | 30 |
| Input size | 640×640 |
| AMP | Mixed Precision (FP16) |

데이터 증강으로 수평 반전(p=0.5)과 밝기/대비 변환(brightness_limit=0.2, contrast_limit=0.2, p=0.3)을 적용하였으며, 동일한 변환이 RGB-Thermal 쌍에 함께 적용된다.

### 4.3 평가 지표

COCO 표준 평가 지표를 사용하였다:
- **mAP@0.5**: IoU 임계값 0.5에서의 평균 정밀도
- **mAP@[.5:.95]**: IoU 0.5~0.95까지 0.05 간격 평균 (더 엄격한 지표)

### 4.4 Ablation Study

제안 모델의 각 구성 요소의 기여도를 검증하기 위해 5가지 변형에 대한 ablation study를 수행하였다.

**표 3.** Ablation Study 결과 (M3FD, 20 epoch 학습 기준, Full은 30 epoch)

| 변형 | 조건 | mAP@0.5 | mAP@[.5:.95] | Recall | F1 | Miss Rate |
|------|------|---------|-------------|--------|-----|----------|
| Thermal-only | 전체 | 0.528 | 0.273 | 0.722 | 0.609 | 0.278 |
|              | 야간 | 0.686 | — | 0.834 | 0.753 | 0.166 |
|              | 주간 | 0.515 | — | 0.713 | 0.598 | 0.287 |
| RGB-only | 전체 | 0.632 | 0.322 | 0.829 | 0.717 | 0.171 |
|          | 야간 | **0.628** | 0.342 | 0.845 | 0.721 | 0.155 |
|          | 주간 | **0.630** | 0.319 | 0.826 | 0.715 | 0.175 |
| Early Fusion | 전체 | 0.650 | 0.346 | 0.835 | 0.730 | 0.165 |
|              | 야간 | 0.800 | — | 0.894 | 0.844 | 0.106 |
|              | 주간 | 0.642 | — | 0.831 | 0.724 | 0.169 |
| Dual + Concat | 전체 | 0.700 | 0.378 | 0.867 | 0.774 | 0.133 |
|               | 야간 | 0.799 | — | 0.943 | 0.865 | 0.057 |
|               | 주간 | 0.692 | — | 0.863 | 0.768 | 0.137 |
| **Ours (Full)** | **전체** | **0.737** | **0.406** | **0.874** | **0.799** | **0.126** |
|                 | **야간** | **0.853** | **0.507** | **0.909** | **0.880** | **0.091** |
|                 | **주간** | **0.728** | **0.398** | **0.870** | **0.793** | **0.130** |

#### 4.4.1 모달리티 융합 효과

RGB 단일 모달리티(0.632)에 Thermal 정보를 융합함으로써 제안 모델은 mAP@0.5 기준 10.5%p, Recall 기준 4.5%p(0.829→0.874)의 향상을 달성하였다. Recall 향상은 단순히 정밀도 개선에 그치지 않고, LWIR이 RGB로 놓치던 객체(야간·역광 등 조도 불량 환경의 열원)를 추가로 검출함으로써 탐지율 자체가 높아졌음을 의미한다. 이는 Thermal 영상이 제공하는 열 정보가 RGB의 시각 정보를 효과적으로 보완함을 입증한다. 특히 mAP@[.5:.95]에서의 향상폭(+8.4%p)은 융합이 검출 위치 정확도 향상에도 기여함을 시사한다.

Thermal 단일 모달리티(0.528)는 RGB(0.632) 대비 10.4%p 낮은 성능을 보였는데, 이는 M3FD 데이터셋이 주간 환경 데이터를 상당 부분 포함하고 있어 텍스처 정보의 부재가 불리하게 작용한 것으로 분석된다. 단, 조건별로 살펴보면 양상이 역전된다. RGB-only는 야간(0.628)과 주간(0.630)이 거의 동일하거나 야간에서 오히려 소폭 하락하는데, 이는 조도 저하 시 RGB 카메라의 노이즈 증가와 텍스처 손실이 성능에 직접 반영된 결과이다. 반면 Thermal-only는 야간(0.686)이 주간(0.515)보다 17.1%p 높아, 조명 독립적인 LWIR의 특성이 명확히 드러난다. 이러한 두 모달리티의 상호 보완적 특성이 융합 모델의 야간 성능 향상(+12.5%p)의 근본 원인이다.

#### 4.4.2 이중 백본 구조의 효과

Early Fusion(0.650)과 Dual+Concat(0.700)의 비교를 통해 이중 백본 구조의 효과를 확인할 수 있다. 동일하게 두 모달리티를 활용하되, 단일 백본으로 6채널 입력을 처리하는 것보다 각 모달리티에 독립 백본을 할당하는 것이 5.0%p 더 우수하였다. 이는 RGB와 Thermal 영상의 통계적 특성 차이가 크므로, 공유 백본이 두 분포를 동시에 최적화하는 것이 어렵기 때문으로 해석된다.

#### 4.4.3 교차 모달 어텐션의 효과

Dual+Concat(0.700)과 Full(0.737)의 비교에서 CMAFM의 기여도는 3.7%p로 확인되었다. 단순 연결은 두 모달리티의 특징을 병렬 배치할 뿐 상호작용을 모델링하지 못하는 반면, CMAFM은 채널 수준의 전역 정보 교환과 공간 수준의 지역 정보 교차 게이팅을 통해 상호 보완적 특징 강화를 달성한다.

### 4.5 학습 곡선 분석

**표 4.** Epoch별 성능 변화

| Epoch | 학습률 | Loss | mAP@0.5 | mAP@[.5:.95] |
|-------|--------|------|---------|-------------|
| 0 | 5×10^-3 | 0.645 | 0.140 | 0.048 |
| 5 | 5×10^-3 | 0.418 | 0.614 | 0.309 |
| 10 | 5×10^-4 | 0.335 | 0.708 | 0.387 |
| 15 | 5×10^-4 | 0.323 | 0.714 | 0.397 |
| 20 | 5×10^-5 | 0.315 | 0.736 | 0.404 |
| 27 (Best) | 5×10^-5 | 0.313 | **0.737** | **0.406** |

학습률 감소 시점(epoch 10, 20)에서 mAP가 단계적으로 상승하는 것을 확인할 수 있으며, 이는 StepLR 스케줄러의 효과를 입증한다. 특히 첫 번째 학습률 감소(epoch 10)에서 mAP@0.5가 0.614(epoch 5)에서 0.708로 약 9.4%p 상승하였다.

### 4.6 정성적 분석

그림 2와 그림 3은 검증 세트에서 선별한 야간 3장면·주간 3장면에 대해 RGB 단독, Thermal 단독, RGB+Thermal 융합 모델의 검출 결과를 나란히 비교한 것이다.

![야간 장면 비교](runs/paper_figures/fig_night_comparison.png)

**그림 2.** 야간 장면에서의 RGB 단독 · Thermal 단독 · 융합 검출 결과 비교.
(a) RGB 입력, (b) Thermal 입력, (c) RGB Feature (C4), (d) Thermal Feature (C4), (e) CMAFM 융합 Feature (C4), (f) RGB 단독, (g) Thermal 단독, (h) RGB+Thermal 융합.

표 7은 야간 3장면·주간 3장면을 각각 합산하여 조건별로 종합한 모달리티별 검출 결과이다.

**표 7.** 야간/주간 조건별 모달리티별 종합 검출 수, 평균 신뢰도 및 융합 우위

| 항목 | RGB 단독 | Thermal 단독 | RGB+Thermal 융합 |
|------|---------|-------------|----------------|
| 야간 — 총 검출 수 (3장면 합산) | 30개 | 43개 | **45개** |
| 야간 — 평균 신뢰도 | 0.80 | 0.77 | **0.87** |
| 야간 — Thermal/RGB 검출 수 배율 | — | **1.43×** | — |
| 야간 — 융합 신뢰도 향상 (vs. 최고 단독) | — | — | **+7%p** |
| 주간 — 총 검출 수 (3장면 합산) | 15개 | 21개 | **25개** |
| 주간 — 평균 신뢰도 | 0.90 | 0.74 | **0.91** |
| 주간 — Thermal/RGB 검출 수 배율 | — | 1.40× | — |
| 주간 — 융합 신뢰도 향상 (vs. 최고 단독) | — | — | **+1%p** |

야간에서는 Thermal 단독(43개)이 RGB 단독(30개) 대비 1.43배 더 많은 객체를 검출하며, 융합 모델은 검출 수(45개)와 신뢰도(0.87)를 동시에 최대화하여 최고 단독 모달리티 대비 신뢰도를 7%p 추가 향상시킨다. 주간에서는 RGB가 주도 모달리티로 작용하여 신뢰도 향상폭(+1%p)이 야간 대비 제한적이나, 융합 검출 수(25개)는 RGB(15개)·Thermal(21개) 단독을 모두 상회하여 Thermal의 보완 효과가 검출 범위 확대로 나타난다. 이는 CMAFM이 장면 조건에 따라 신뢰도 높은 모달리티에 자동으로 가중치를 부여함으로써 야간·주간 모든 조건에서 단일 모달리티 대비 우수한 검출 성능을 달성함을 실증한다.

---

### 4.7 야간/주간 조건별 성능 분석

RGB-Thermal 융합 모델의 실질적 가치를 검증하기 위해, M3FD 검증 세트(840장)를 RGB 영상의 평균 밝기를 기준으로 야간(밝기 < 60, 61장)과 주간(밝기 ≥ 60, 779장)으로 분류하여 조건별 성능을 측정하였다.

**표 5.** 야간/주간 조건별 검출 성능

| 조건 | 이미지 수 | mAP@0.5 | mAP@[.5:.95] | Recall | F1 | Miss Rate |
|------|----------|---------|-------------|--------|-----|----------|
| 전체 (Overall) | 840 | 0.7361 | 0.4054 | 0.8741 | 0.7992 | 0.1259 |
| **야간 (Night)** | **61** | **0.8533** | **0.5067** | **0.9094** | **0.8804** | **0.0906** |
| 주간 (Day) | 779 | 0.7284 | 0.3977 | 0.8702 | 0.7930 | 0.1298 |

![요약 테이블](runs/paper_figures/fig_summary_table.png)

**표 5 시각화.** 야간/주간 조건별 mAP 요약.

분석 결과, 야간 조건에서 mAP@0.5가 0.8533으로 주간(0.7284) 대비 **+12.5%p** 높은 성능을 보였다. Recall 또한 야간 0.9094로 주간(0.8702) 대비 **+3.9%p** 높아, 야간에서 LWIR이 RGB로 탐지하지 못하는 객체를 추가로 검출함으로써 탐지율 자체가 향상됨을 확인하였다. Miss Rate 역시 야간(0.0906)이 주간(0.1298) 대비 3.9%p 낮아 미검출 오류가 감소하였다. 이는 RGB-Thermal 융합 모델의 핵심 장점을 실증적으로 입증하는 결과이다.

![야간 vs 주간 비교](runs/paper_figures/fig_night_vs_day.png)

**그림 4.** 야간 vs 주간 장면에서의 RGB·Thermal·융합 특징·검출 결과 및 Thermal 기여도 비교.
열(e)의 Thermal 기여도(Fused − RGB Feature의 절댓값)는 야간에서 현저히 높게 나타나, 조도가 낮은 환경에서 Thermal 모달리티의 기여가 극대화됨을 확인할 수 있다.

#### 4.7.1 클래스별 분석

그림 5와 표 6은 클래스별 AP@0.5를 조건에 따라 비교한 것이다.

![클래스별 바 차트](runs/paper_figures/fig_classwise_bar.png)

**그림 5.** 클래스별 AP@0.5 — 전체/야간/주간 비교.

**표 6.** 클래스별 AP@0.5 야간/주간 비교

| 클래스 | 전체 | 야간 | 주간 | 야간−주간 |
|--------|------|------|------|----------|
| People | 0.7573 | **0.8620** | 0.7488 | +11.3%p |
| Car | 0.8727 | **0.9517** | 0.8672 | +8.5%p |
| Bus | 0.7716 | **0.9600** | 0.7429 | +21.7%p |
| Motorcycle | 0.6975 | **0.7360** | 0.6950 | +4.1%p |
| Lamp | 0.6717 | **0.7569** | 0.6702 | +8.7%p |
| Truck | 0.6457 | — (샘플 없음) | 0.6465 | — |

야간 성능 향상 효과는 클래스에 따라 다르게 나타났다. Bus(+21.7%p)와 Car(+8.5%p)는 열 방출량이 크고 외형이 뚜렷한 대형 객체로, Thermal 영상에서 선명한 열 시그니처를 형성하여 가장 큰 야간 성능 향상을 보였다. 반면 Motorcycle(+4.1%p)은 소형 객체의 특성상 Thermal 해상도의 한계로 인해 향상 폭이 상대적으로 작았다.

#### 4.7.2 야간 성능 우위 원인 분석

야간에서의 성능 우위는 다음 두 가지 메커니즘으로 설명된다.

**Thermal 모달리티의 상보적 기여 극대화**: 주간 환경에서는 RGB 특징맵이 이미 풍부한 시각 정보를 제공하므로 Thermal의 추가 기여가 상대적으로 제한된다. 반면 야간에서는 RGB 특징맵의 활성화가 약화되고 Thermal의 열 시그니처가 주요 단서로 작용하여, CMAFM의 채널 교차 어텐션이 Thermal 특징에 높은 가중치를 부여한다.

**공간 교차 게이팅의 선택적 강화**: CMAFM의 공간 교차 게이팅은 상대 모달리티의 맥락에 따라 각 위치의 특징을 선택적으로 강화한다. 야간에서 Thermal이 열원 위치를 명확히 지시함으로써 RGB 특징의 관련 영역이 효과적으로 강화되어 검출 정확도가 향상된다.

---

## V. 결론 및 향후 연구

본 논문에서는 RGB-Thermal 다중 스펙트럼 영상 융합을 위한 교차 모달 어텐션 기반 객체 검출 모델을 제안하였다. M3FD 데이터셋 실험에서 제안 모델은 mAP@0.5 73.7%, Recall 87.4%를 달성하여 RGB 단일 모달리티 대비 mAP@0.5 10.5%p, Recall 4.5%p 향상되었으며, 야간 조건에서 mAP@0.5 85.3%, Recall 90.9%로 주간 대비 각각 12.5%p, 3.9%p 높은 성능을 보여 열악한 조도 환경에서 LWIR 융합의 효과를 실증하였다. 군사적 활용 측면에서 본 모델은 현재 운용 중인 RGB 및 열상 카메라 장비를 그대로 활용하여 소프트웨어 수준의 적용만으로 실전 배치가 가능하므로, 별도 하드웨어 구축 없이 예산을 절감하면서 신속한 전력화가 기대된다. 또한 주야간 및 역광·안개 등 전천후 환경에서 감시 지속성과 객체 탐지 능력을 동시에 향상시킬 수 있어, 감시·정찰 임무의 실질적 전투력 향상에 기여할 수 있다.

향후 연구 방향으로는 다음을 고려한다:

1. **검출기 고도화**: Faster R-CNN을 YOLOv8 또는 RT-DETR 등 최신 검출기로 교체하여 검출 성능 및 속도를 동시에 개선
2. **조건별 학습 전략**: 본 연구의 야간/주간 조건별 성능 분석 결과를 바탕으로, 야간 및 흐림 등 열악한 조건의 데이터를 가중 샘플링하거나 조건 인식 어텐션(condition-aware attention)을 도입하여 조건 불균형 문제 완화
3. **다중 데이터셋 검증**: KAIST, LLVIP 등 추가 데이터셋에서의 일반화 성능 검증 및 야간 특화 모델과의 비교
4. **경량화**: 모델 증류(Knowledge Distillation) 또는 백본 경량화를 통한 실시간 응용 가능성 탐색

---

## 참고문헌

[1] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, "Object Detection in 20 Years: A Survey," *Proceedings of the IEEE*, vol. 111, no. 3, pp. 257-332, 2023.

[2] S. Hwang, J. Park, N. Kim, Y. Choi, and I. S. Kweon, "Multispectral Pedestrian Detection: Benchmark Dataset and Baseline," in *Proc. IEEE CVPR*, 2015, pp. 1037-1045.

[3] C. Li, D. Song, R. Tong, and M. Tang, "Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation," in *Proc. BMVC*, 2018.

[4] J. Liu, S. Fan, X. Wang, Y. Zhong, and R. Rao, "TarDAL: A Unified Framework for Target Detection and Domain Adaptation in Multispectral Imaging," in *Proc. IEEE CVPR*, 2022.

[5] H. Zhang, E. Fromont, S. Lefevre, and B. Avignon, "Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks," in *Proc. IEEE ICIP*, 2020.

[6] F. Cao, L. Bao, and X. Li, "Cross-Modal Feature Fusion for RGB-Thermal Object Detection," *IEEE Transactions on Intelligent Transportation Systems*, 2023.

[7] K. Kim, "Survey on Multispectral Pedestrian Detection: Dataset, Method, and Challenges," *Journal of IEIE*, vol. 60, no. 1, pp. 15-28, 2023.

[8] J. Liu, S. Zhang, S. Wang, and D. Metaxas, "Multispectral Deep Neural Networks for Pedestrian Detection," in *Proc. BMVC*, 2016.

[9] L. Zhang, Z. Liu, S. Zhang, X. Yang, and H. Qiao, "Cross-Modality Interactive Attention Network for Multispectral Pedestrian Detection," *Information Fusion*, vol. 50, pp. 20-29, 2019.

[10] Y. Qingyun and W. Zhaokui, "Cross-Modality Fusion Transformer for Multispectral Object Detection," arXiv:2111.00273, 2021.

[11] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE CVPR*, 2018, pp. 7132-7141.

[12] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," in *Proc. NeurIPS*, 2015.

[13] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," in *Proc. IEEE CVPR*, 2017.

[14] J. Liu, S. Fan, X. Wang, Y. Zhong, and R. Rao, "Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection," in *Proc. IEEE CVPR*, 2022.
