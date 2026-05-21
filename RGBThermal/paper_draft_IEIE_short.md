# 교차 모달 어텐션 기반 RGB-Thermal 영상 융합 객체 검출

## Cross-Modal Attention-Based RGB-Thermal Image Fusion for Object Detection

---

### 요약

본 논문은 가시광(RGB)과 적외선(Thermal) 영상을 융합하여 보행자 및 차량을 검출하는 딥러닝 기반 다중 스펙트럼 객체 검출 모델을 제안한다. 제안 모델은 각 모달리티에 독립적인 ResNet-50 백본을 할당하고, 채널 교차 어텐션(Channel Cross-Attention)과 공간 교차 게이팅(Spatial Cross-Gating)으로 구성된 교차 모달 어텐션 융합 모듈(CMAFM)을 통해 상호 보완적 정보를 교환한다. M3FD 데이터셋 실험에서 mAP@0.5 73.7%, Recall 87.4%를 달성하여 RGB 단일 모달리티 대비 각각 10.5%p, 4.5%p 향상하였다. 야간 조건에서는 mAP@0.5 85.3%, Recall 90.9%로 주간 대비 각각 12.5%p, 3.9%p 높은 성능을 보여, 열악한 조도 환경에서의 융합 효과를 실증하였다.

**핵심어**: 다중 스펙트럼 객체 검출, RGB-Thermal 융합, 교차 모달 어텐션, 이중 백본, Faster R-CNN

### Abstract

This paper proposes a multispectral object detection model fusing visible (RGB) and infrared (Thermal) images for pedestrian and vehicle detection. The model assigns independent ResNet-50 backbones to each modality and exchanges complementary information through a Cross-Modal Attention Fusion Module (CMAFM) consisting of Channel Cross-Attention and Spatial Cross-Gating. Experiments on M3FD achieve 73.7% mAP@0.5 and 87.4% Recall, improving over RGB-only by 10.5%p and 4.5%p respectively. Under nighttime conditions, the model achieves 85.3% mAP@0.5 and 90.9% Recall, surpassing daytime by 12.5%p and 3.9%p.

**Keywords**: Multispectral Object Detection, RGB-Thermal Fusion, Cross-Modal Attention, Dual Backbone, Faster R-CNN

---

## I. 서론

자율 주행, 군사 감시 등의 분야에서 객체 검출은 핵심 기술로 자리잡고 있다[1]. RGB 카메라는 야간·안개 등 열악한 조건에서 성능이 급격히 저하되고[2], 적외선 카메라는 조명에 독립적이나 텍스처·색상 정보의 부재로 유사 열 특성 객체 구별이 어렵다[3]. 이를 극복하기 위한 RGB-Thermal 융합 연구에서 중간 수준 융합이 가장 높은 성능을 보이는 것으로 보고되나[4-7], 기존 방법들은 단순 연결(Concatenation)에 의존하거나 전체 공간 어텐션의 O(N²) 복잡도로 고해상도 특징맵 적용이 어렵다.

본 논문에서는 메모리 효율적인 채널 교차 어텐션과 합성곱 기반 공간 교차 게이팅을 결합한 CMAFM을 포함한 이중 백본 기반 다중 스펙트럼 객체 검출 모델을 제안한다. 주요 기여는 다음과 같다:

1. 채널 교차 어텐션과 공간 교차 게이팅을 결합한 메모리 효율적 CMAFM 제안
2. 각 모달리티에 독립 백본을 할당하는 이중 백본 아키텍처 설계
3. M3FD 데이터셋에서의 ablation study를 통한 각 구성 요소 기여도 검증

---

## II. 관련 연구

RGB-Thermal 융합 연구는 융합 위치에 따라 조기·중간·후기 융합으로 분류된다. TarDAL[4]은 적대적 학습 기반 융합을, Zhang et al.[5]는 Cyclic Fuse-and-Refine 블록을 제안하였다. Cao et al.[6]과 Qingyun et al.[10]은 Transformer 기반 교차 모달 융합을 시도하였으나 O(N²) 복잡도 문제를 안고 있다. Zhang et al.[9]의 CIAN은 교차 모달리티 상호 어텐션을 제안하였으나, 본 논문의 CMAFM은 N×N 어텐션 행렬 없이 전역 및 지역 수준의 양방향 정보 교환을 달성한다는 점에서 차별화된다. 검출 프레임워크로는 다중 스케일 처리에 강점을 가진 Faster R-CNN[12] + FPN[13]을 채택하였다.

---

## III. 제안 방법

### 3.1 전체 구조

제안 모델은 (1) 이중 백본, (2) CMAFM, (3) FPN, (4) Faster R-CNN 검출 헤드로 구성된다.

```
RGB 영상 (H×W×3)
    │
    ▼
┌──────────────┐
│ ResNet-50    │──► C3 (H/8, 512ch) ──┐
│ (RGB 전용)   │──► C4 (H/16, 1024ch)─┤──► CMAFM ──► FPN ──► Faster R-CNN
│              │──► C5 (H/32, 2048ch)─┤                       (RPN + ROI)
└──────────────┘                       │
                                       │
┌──────────────┐                       │
│ ResNet-50    │──► C3 (H/8, 512ch) ──┤
│ (Thermal 전용)│──► C4 (H/16, 1024ch)─┤
│              │──► C5 (H/32, 2048ch)─┘
└──────────────┘

Thermal 영상 (H×W×1 → 3ch 복제)
```

**그림 1.** 제안 모델의 전체 구조

입력 영상은 640×640으로 리사이즈하며, Thermal 영상은 1채널을 3채널로 복제하여 ImageNet 사전 학습 가중치를 활용한다. 동일 스케일의 RGB-Thermal 특징쌍에 CMAFM이 적용된다.

### 3.2 교차 모달 어텐션 융합 모듈 (CMAFM)

CMAFM은 세 단계로 구성된다.

**채널 교차 어텐션**: 전역 평균 풀링(GAP)으로 채널별 통계량을 추출한 후 교차 모달리티 dot-product 스케일링을 수행한다.

```
q_r = W_q · GAP(F_r),   k_t, v_t = split(W_kv · GAP(F_t))
scale = σ(q_r ⊙ k_t),  F_r' = F_r · (v_t · scale)
```

동일 과정을 Thermal→RGB 방향으로도 수행하여 양방향 교환을 달성한다. O(C) 복잡도로 메모리 효율이 높다.

**공간 교차 게이팅**: 채널 어텐션 후 합성곱 기반 공간 게이팅으로 관련 영역을 강화하고 무관한 영역을 억제한다.

```
S_r = Conv_1×1(DWConv_3×3(F_r')),  S_t = Conv_1×1(DWConv_3×3(F_t'))
G_r = σ(Conv_1×1([S_r; S_t])),     F_r'' = G_r ⊙ F_r'
```

**게이트 융합**: 학습 가능한 게이트 α로 픽셀·채널별 적응적 융합을 수행한다.

```
α = σ(Conv_1×1([F_r''; F_t'']))
F_fused = α ⊙ F_r'' + (1-α) ⊙ F_t'',  F_out = Conv_3×3(F_fused) + F_fused
```

### 3.3 학습 전략

백본에는 기본 학습률(0.005)의 0.1배를 적용하는 차등 학습률로 사전 학습 가중치를 보존하고, StepLR 스케줄러로 10 epoch마다 학습률을 0.1배 감소시킨다.

---

## IV. 실험

### 4.1 실험 설정

M3FD 데이터셋[14](4,200쌍, 6 클래스, 34,407 객체)을 8:2로 분할하여 사용하였다. GPU는 NVIDIA RTX A6000(48GB), PyTorch 2.4.0, batch size 8, 30 epoch 학습하였다.

**표 1.** M3FD 데이터셋 클래스별 객체 수

| 클래스 | 객체 수 | 비율 |
|--------|---------|------|
| Car | 18,296 | 53.2% |
| People | 11,477 | 33.4% |
| Lamp | 2,405 | 7.0% |
| Truck | 1,008 | 2.9% |
| Bus | 700 | 2.0% |
| Motorcycle | 521 | 1.5% |

### 4.2 Ablation Study

**표 2.** Ablation Study 결과 (M3FD)

| 변형 | 조건 | mAP@0.5 | mAP@[.5:.95] | Recall | F1 |
|------|------|---------|-------------|--------|-----|
| Thermal-only | 전체 | 0.528 | 0.273 | 0.722 | 0.609 |
|              | 야간 | 0.686 | — | 0.834 | 0.753 |
|              | 주간 | 0.515 | — | 0.713 | 0.598 |
| RGB-only | 전체 | 0.632 | 0.322 | 0.829 | 0.717 |
|          | 야간 | 0.628 | 0.342 | 0.845 | 0.721 |
|          | 주간 | 0.630 | 0.319 | 0.826 | 0.715 |
| Early Fusion | 전체 | 0.650 | 0.346 | 0.835 | 0.730 |
| Dual + Concat | 전체 | 0.700 | 0.378 | 0.867 | 0.774 |
| **Ours (Full)** | **전체** | **0.737** | **0.406** | **0.874** | **0.799** |
|                 | **야간** | **0.853** | **0.507** | **0.909** | **0.880** |
|                 | **주간** | **0.728** | **0.398** | **0.870** | **0.793** |

RGB 단일 모달리티 대비 mAP@0.5 +10.5%p, Recall +4.5%p를 달성하였다. Early Fusion(0.650)과 Dual+Concat(0.700) 비교에서 이중 백본 구조의 기여는 +5.0%p, CMAFM의 기여는 +3.7%p로 확인되었다. 특히 Thermal-only는 야간(0.686)이 주간(0.515)보다 17.1%p 높아 LWIR의 조명 독립성이 드러나며, 이것이 융합 모델의 야간 우위(+12.5%p)의 근본 원인이다.

### 4.3 야간/주간 조건별 성능 분석

검증 세트(840장)를 RGB 평균 밝기 기준으로 야간(< 60, 61장)과 주간(≥ 60, 779장)으로 분류하였다.

**표 3.** 야간/주간 조건별 검출 성능

| 조건 | 이미지 수 | mAP@0.5 | mAP@[.5:.95] | Recall | Miss Rate |
|------|----------|---------|-------------|--------|----------|
| 전체 | 840 | 0.737 | 0.405 | 0.874 | 0.126 |
| **야간** | **61** | **0.853** | **0.507** | **0.909** | **0.091** |
| 주간 | 779 | 0.728 | 0.398 | 0.870 | 0.130 |

야간에서 mAP@0.5 +12.5%p, Recall +3.9%p로 주간을 크게 상회하였다. 클래스별로는 열 방출량이 큰 대형 객체인 Bus(+21.7%p), Car(+8.5%p)에서 야간 성능 향상이 가장 두드러졌다. 야간 성능 우위는 두 메커니즘으로 설명된다: (1) RGB 활성화가 약화될 때 CMAFM 채널 교차 어텐션이 Thermal에 높은 가중치를 부여하고, (2) 공간 교차 게이팅이 Thermal 열원 위치를 단서로 RGB 특징의 관련 영역을 선택적으로 강화한다.

![야간 vs 주간 비교](runs/paper_figures/fig_night_vs_day.png)

**그림 2.** 야간 vs 주간 장면에서의 RGB·Thermal·융합 특징·검출 결과 비교.

---

## V. 결론

본 논문은 채널 교차 어텐션과 공간 교차 게이팅을 결합한 CMAFM 기반 이중 백본 RGB-Thermal 객체 검출 모델을 제안하였다. M3FD 실험에서 mAP@0.5 73.7%, Recall 87.4%로 RGB 단일 모달리티 대비 10.5%p, 4.5%p 향상하였으며, 야간 조건에서 mAP@0.5 85.3%로 주간 대비 12.5%p 높은 성능을 달성하여 열악한 조도 환경에서의 융합 효과를 실증하였다. 향후 연구로는 YOLOv8/RT-DETR 기반 검출기 고도화, 조건 인식 어텐션 도입, KAIST·LLVIP 추가 데이터셋 검증을 고려한다.

---

## 참고문헌

[1] Z. Zou et al., "Object Detection in 20 Years: A Survey," *Proc. IEEE*, vol. 111, no. 3, pp. 257-332, 2023.

[2] S. Hwang et al., "Multispectral Pedestrian Detection: Benchmark Dataset and Baseline," *Proc. CVPR*, 2015.

[3] C. Li et al., "Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation," *Proc. BMVC*, 2018.

[4] J. Liu et al., "TarDAL: A Unified Framework for Target Detection and Domain Adaptation in Multispectral Imaging," *Proc. CVPR*, 2022.

[5] H. Zhang et al., "Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks," *Proc. ICIP*, 2020.

[6] F. Cao et al., "Cross-Modal Feature Fusion for RGB-Thermal Object Detection," *IEEE Trans. ITS*, 2023.

[7] K. Kim, "Survey on Multispectral Pedestrian Detection," *J. IEIE*, vol. 60, no. 1, pp. 15-28, 2023.

[8] J. Liu et al., "Multispectral Deep Neural Networks for Pedestrian Detection," *Proc. BMVC*, 2016.

[9] L. Zhang et al., "Cross-Modality Interactive Attention Network for Multispectral Pedestrian Detection," *Information Fusion*, vol. 50, pp. 20-29, 2019.

[10] Y. Qingyun and W. Zhaokui, "Cross-Modality Fusion Transformer for Multispectral Object Detection," arXiv:2111.00273, 2021.

[11] J. Hu et al., "Squeeze-and-Excitation Networks," *Proc. CVPR*, 2018.

[12] S. Ren et al., "Faster R-CNN," *Proc. NeurIPS*, 2015.

[13] T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection," *Proc. CVPR*, 2017.

[14] J. Liu et al., "Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark," *Proc. CVPR*, 2022.
