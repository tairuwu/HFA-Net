# HFA-Net: Hierarchical Feature Aggregation Network for Micro-Expression Recognition
Published in Complex & Intelligent Systems

## Abstract
Micro-expressions (MEs) are unconscious and involuntary reactions that genuinely reflect an individual’s inner emotional state, making them valuable in the fields of emotion analysis and behavior recognition. MEs are characterized by subtle changes within specific facial action units, and effective feature learning and fusion tailored to these characteristics still require in-depth research. To address this challenge, this paper proposes a novel hierarchical feature aggregation network (HFA-Net). In the local branch, the multi-scale attention (MSA) block is proposed to capture subtle facial changes and local information. The global branch introduces the retentive meet transformers (RMT) block to establish dependencies between holistic facial features and structural information. Considering that single-scale features are insufficient to fully capture the subtleties of MEs, a multi-level feature aggregation (MLFA) module is proposed to extract and fuse features from different levels across the two branches, preserving more comprehensive feature information. To enhance the representation of key features, an adaptive attention feature fusion (AAFF) module is designed to focus on the most useful and relevant feature channels. Extensive experiments conducted on the SMIC, CASME II, and SAMM benchmark databases demonstrate that the proposed HFA-Net outperforms current state-of-the-art methods. Additionally, ablation studies confirm the superior discriminative capability of HFA-Net when learning feature representations from limited ME samples. Our code is publicly available at https://github.com/tairuwu/HFA-Net.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhang2025hfa,
  author    = {Zhang, Meng and Yang, Wenzhong and Wang, Liejun and Wu, Zhonghua and Chen, Danny},
  title     = {HFA-Net: hierarchical feature aggregation network for micro-expression recognition},
  journal   = {Complex \& Intelligent Systems},
  year      = {2025},
  volume    = {11},
  number    = {3},
  pages     = {169},
  doi       = {10.1007/s40747-025-01804-0},
  url       = {https://doi.org/10.1007/s40747-025-01804-0}
}
``` 

