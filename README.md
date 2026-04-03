# DA-SSMA: A Few-Shot Radio Frequency Fingerprint Identification Scheme Based on Dual Attention Mechanism and Semi-Supervised Metric Adversarial Learning
# Paper
The code corresponds to the paper [https://ieeexplore.ieee.org/document/11268505](https://ieeexplore.ieee.org/document/11268505)

X. Cao, W. Tan, Q. Gao, F. Zhang and C. Li, "DA-SSMA: A Few-Shot Radio Frequency Fingerprint Identification Scheme Based on Dual Attention Mechanism and Semi-Supervised Metric Adversarial Learning," in IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 5092-5105, 2026, doi: 10.1109/TCCN.2025.3637113.
# Requirement
python == 3.9

numpy==2.0.2

torch==2.3.1

pandas==2.2.2

matplotlib==3.9.0

scikit-learn==1.5.1

fvcore==0.1.5.post20221221

seaborn==0.13.2

PyYAML==6.0.1

torchvision==0.20.1

utils==1.0.2

thop==0.1.1.post2209072238
# Abstract
Radio frequency fingerprint identification (RFFI) is a technology that uses the characteristics of radio signals to identify devices or users. Deep learning (DL) has been widely applied in RFFI due to its superior extraction and classification capabilities. However, existing DL-based RFFI methods rely on a large amount of labeled data and have insufficient ability to characterize signal features in few-shot scenarios, leading to limited generalization performance and poor recognition capabilities in low SNR environments. Therefore, we proposes a few-shot RFFI (FS-RFFI) method based on the fusion of dual attention mechanism and semi-supervised metric adversarial learning (DA-SSMA). Specifically, data augmentation is used to improve the anti-interference ability and generalization performance of the model. Through the semi-supervised metric adversarial learning framework and training with unlabeled data, the robustness and generalization ability of the model to few-shot are improved. Then, a dual attention mechanism is adopted for feature recalibration, which enhances the model’s ability to capture key signal features, and an objective function is designed to extract the discriminative and generalized semantic features of radio signals. The proposed DA-SSMA method is verified on the real-world automatic-dependent surveillance–broadcast (ADS-B) dataset and the Wi-Fi dataset. The experimental results show that the proposed method has higher recognition accuracy than the existing methods. Specifically, when the proportion of labeled data is 10%, the recognition accuracy of ADS-B reached 95.90%, and that of Wi-Fi reaches 98.13%. Under the low SNR of -5db, ADS-B still has a recognition accuracy of 84.60%.
# Framework of DA-SSMA
## framework of DA-SSMA
![framework of DA-SSMA](https://i-blog.csdnimg.cn/direct/8c18e62d0e1845da8df606d1adf5b5b5.png#pic_center)
## framework of DA
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/26531bfd55554beba3a91cf2cb8b11dc.png#pic_center)
## framework of SSMA
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1752b9d7362640b7b0a251799983b8be.png#pic_center)
# Dataset and Experimental Setup
The model of the proposed DA-SSMA method is based on PyTorch, and the simulation platform is GTX 1080Ti. We used the ADS-B and Wi-Fi datasets to verify our model. The detailed signal acquisition process can be found in the papers [43] and [44]. We use 10 ADS-B signal datasets for simulation, each with a sample length of 4800, a total of 3080 samples, and 1000 test samples.The Wi-Fi dataset was collected from 16 USRP X310 radios, each with a sample length of 6000, a total of 4800 samples, and 1600 test samples. We conduct experiments based on the ratio of the number of labeled training samples to the number of samples as {5\%, 10\%, 20\%} to evaluate the recognition performance of the proposed DA-SSMA method. The network architecture parameters of the model are shown in Table \ref{dassma_arch}, and during the model training process, the training data, validation data and test data are distributed in the ratio of 8:1:1. We used the Adam optimizer for parameter training, with a learning rate of 0.001 and a perturbation intensity of VAT of 1.0. The model was trained for 300 epochs with a batch size of 32.

[43]T. Ya, L. Yun, Z. Haoran, W. Yu, G. Guan, and M. Shiwen, “Large-scale real-world radio signal recognition with deep learning,” Chinese Journal of Aeronautics, vol. 35, no. 9, pp. 35–48, 2022.

[44]K. Sankhe, M. Belgiovine, F. Zhou, S. Riyaz, S. Ioannidis, and K. Chowdhury, “Oracle: Optimized radio classification through convolutional neural networks,” in Proceedings of the IEEE Conference onComputer Communications (INFOCOM), 2019, pp. 370–378.
# Classification Accuracy

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4020a5bcaba245799d1e2dae0c1fe508.png#pic_center)
# Visualization of Semantic Features by Different Methods
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c29d8a847bcb4869bee733d430c09b9e.png#pic_center)
# Robustness Verification of DA-SSMA
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b7e78af5ae34f11945ae06f2a531905.png#pic_center)
# Email
If you have any question, please feel free to contact us by e-mail (gs.xcao23@gzu.edu.cn).

