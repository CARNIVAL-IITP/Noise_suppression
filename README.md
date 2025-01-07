# Speech_enhancement
본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상화의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 잡음제거의 1차년도 (2021), 2차년도 (2022) 코드입니다.
- 1차년도 모델의 경우 Microsoft에서 개최한 DNS-Shallenge의 DB로 훈련하였으며 총 10,000 set의 training set과 5,000 set의 validation set을 사용했습니다.
- 2차년도 모델의 경우 Sitec 한국어 음성 DB를 사용하여 진행되었습니다.

DCCRN 모델을 기반으로 다채널 모델로 확장 및 minimum statistics algorithm을 이용하여 성능을 개선하는 연구개발을 진행했습니다.1

# Requirement
audioread               2.1.9                              
numpy                   1.22.0               
pesq                    0.0.4                
pypesq                  1.2.4                
scipy                   1.7.2                
SoundFile               0.10.3.post1         
tensorboard             2.10.0               
torch                   1.9.0                
wandb                   0.13.1               

# Training:

    python train.py

# Evluation:

    python inference.py


# Reference:

* Y. Hu, Y. Liu, S. Lv, M. Xing, S. Zhang, Y. Fu, J. Wu, B. Zhang, and L. Xie, “DCCRN: Deep complex convolution recurrent network for phase-aware speech enhancement,” in Proc. Interspeech, 2020, pp. 2472–2476.
* Martin, “Noise power spectral density estimation based on optimal smoothing and minimum statistics,” IEEE Trans. Speech Audio Proc., vol. 9, no. 5, pp. 504–512, 2001.


