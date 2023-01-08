# T-CAD

[A Three-Stage Anomaly Detection Framework for Traffic Videos](https://www.hindawi.com/journals/jat/2022/9463559/)

**Abstract:**

As reported by the United Nations in 2021, road accidents cause 1.3 million deaths and 50 million injuries worldwide each year. Detecting traffic anomalies timely and taking immediate emergency response and rescue measures are essential to reduce casualties, economic losses, and traffic congestion. This paper proposed a three-stage method for video-based traffic anomaly detection. In the first stage, the ViVit network is employed as a feature extractor to capture the spatiotemporal features from the input video. In the second stage, the class and patch tokens are fed separately to the segment-level and video-level traffic anomaly detectors. In the third stage, we finished the construction of the entire composite traffic anomaly detection framework by fusing outputs of two traffic anomaly detectors above with different granularity. Experimental evaluation demonstrates that the proposed method outperforms the SOTA method with 2.07% AUC on the TAD testing overall set and 1.43% AUC on the TAD testing anomaly subset. This work provides a new reference for traffic anomaly detection research.


# Results

## Result of TAD dataset
| Class	| Method | Overall set AUC (%) | Anomaly subset AUC (%) |
|:------|:-------|:----|:----|
| Unsupervised |	Luo et al. [32] |	57.89 |	55.84 |
| | Liu et al. [37]	| 69.13	| 55.38 |
| Weakly supervised	| Sultani et al. [41] |	81.42 |	55.97 |
| | Zhu et al. [45]	| 83.08	| 56.89 |
| | Lv et al. [54]	| 89.64	| 61.66 |
| | Ours	| 91.71 |	63.09 |

## Ablation studies on TAD dataset
| Dataset |	Methods	Recall (%) | Recall (%) | Precision (%) |	F1-score |	AUC (%) |
|:------|:-------|:----|:----|:----|:----|
| Overall set |	T-SAD |	92.16 |	90.17 |	0.9088 | 91.05 |
| | T-CAD | 92.00 | 90.48 | 0.9109 | 91.71 |
| Anomaly subset |	T-SAD |	66.68 |	62.68 |	0.6154 | 62.04 |	
| | T-CAD | 66.62 | 63.15 | 0.6279 | 63.09 |

# Get started

## Training
`python model_SAD_train.py --data_path DATA/TAD/  --epoches 500 --batch_size 4 --save_path parames`

`python model_VAD_train.py --data_path DATA/TAD/  --epoches 500 --batch_size 8 --save_path parames`

## Testing

`python model_test.py --data_path DATA/TAD/  --result_path result/ --SAD_model_path parames/model_SAD_best_auc.pth --VAD_model_path parames/model_VAD_best_auc.pth`

# Citation
If our paper help your work, please cite our papers:
```BibTeX
@ARTICLE{9878251,
  author={Junzhou Chen, Jiancheng Wang, Jiajun Pu, Ronghui Zhang},
  journal={Journal of Advanced Transportation}, 
  title={A Three-Stage Anomaly Detection Framework for Traffic Videos}, 
  year={2022},
  volume={},
  number={},
  pages={},
  doi={10.1155/2022/9463559}}
```

# Acknowledgement

[ViVit](https://github.com/mx-mark/VideoTransformer-pytorch)

Thanks for the great implementations!
