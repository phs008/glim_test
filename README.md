
## Introduction

glim 오픈소스를 분석하되 이미지 기반으로 slam 구현이 가능한지  front-end / back-end 단으로 분석을 진행함.

하지만 해당 오픈소스는 기본적으로 이미지 매칭 이후 keyframe 정의 할수 있도록 기본 descriptor 매칭 시스템 구현이 전혀 안되어있는 상황으로서 

visual slam 의 front 요소중 필수인 keyframe 선정 등에 적합하지 않아
본 개발에서는 visual odometry 관점에서만 구현진행함.

기본 컨샙은 mono odometry (slam) 형태로서 
1. 이미지 간의 feature 를 기반으로 pnp 를 수행
2. pnp 를 이용하여 초기 camera pose 추정
3. 추정된 camera pose frame 이 30개 이상일 경우 한번씩 BA 수행
으로 이뤄진다.

- ***수정및추가된부분:*** src/glim/odometry/odometry_estimation_imgonly.cpp

- ***lib 추가부분:*** BA 를 위한 Ceres-Solver 추가

## 소회
개인적인 의견이지만 해당 라이브러리는 Visual slam 기반 오픈소스로는 매우 부적합하다 판단됨.

해당 라이브러리는 Visual-inertial slam 을 구현할때 일부 내용을 차용해서 가져다 쓸수 있는 구조로는 보임

