# RGB-D 기반 얼굴 주름 검출

RGB 이미지와 iPhone LiDAR Depth 데이터를 활용한 얼굴 주름 영역 검출 및 분할 프로젝트입니다.

## 📌 Project Overview

기존 RGB 이미지 기반 주름 검출의 한계를 보완하기 위해 RGB 이미지와 iPhone LiDAR Depth 데이터를 함께 활용하여 얼굴의 주름 영역을 검출하는 모델을 개발했습니다.

RGB 정보뿐만 아니라 얼굴 표면의 깊이(Depth) 정보를 추가하여 주름의 형태와 깊이 정보를 함께 학습하는 것을 목표로 했습니다.

## 🔍 Project Pipeline

RGB Image + iPhone LiDAR Depth
↓
Data Preprocessing
↓
HRNet Keypoint Detection
↓
HRNet Transfer Learning
↓
Background Removal (rembg)
↓
Depth Edge Detection
↓
Canny Edge Detection
↓
RGB-D Early Fusion
↓
U-Net++ Segmentation
↓
Wrinkle Detection

## 🗂️ Data

- RGB facial images
- iPhone LiDAR Depth data
- LabelMe JSON annotations
- Segmentation masks

※ 개인정보 보호 및 데이터 관리상의 이유로 실제 얼굴 이미지와 원본 데이터는 본 저장소에 포함하지 않았습니다.

## ⚙️ Data Preprocessing

### 1. Background Removal

`rembg`를 이용하여 얼굴 영역 외의 배경을 제거했습니다.

학습 과정과 추론 과정에서 동일한 배경 제거 전처리를 적용하여 데이터 처리 과정의 일관성을 유지했습니다.

### 2. Keypoint Detection

HRNet 기반 Keypoint Detection 모델을 활용하여 얼굴의 주요 영역을 검출했습니다.

또한 Transfer Learning을 적용하여 얼굴 데이터에 적합하도록 모델을 학습했습니다.

### 3. Depth Processing

iPhone LiDAR를 통해 획득한 Depth 데이터를 전처리하고 RGB 이미지와 함께 활용했습니다.

Depth 데이터를 이용하여 얼굴 표면의 3차원적인 형태 정보를 활용할 수 있도록 구성했습니다.

### 4. Depth Edge Detection

Depth 데이터에서 얼굴 표면의 깊이 변화가 발생하는 영역을 확인하기 위해 Depth Edge Detection을 수행했습니다.

또한 Canny Edge Detection을 활용하여 주름 영역의 경계 정보를 추출했습니다.

## 🧠 Model

### RGB Baseline

기존 RGB 이미지 기반 모델을 Baseline으로 설정했습니다.

```text
RGB Image
    ↓
U-Net++
    ↓
Wrinkle Segmentation