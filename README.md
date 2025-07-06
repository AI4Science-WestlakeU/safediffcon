# SafeDiffCon (From Uncertain to Safe: Conformal Adaptation of Diffusion Models for Safe PDE Control) (ICML 2025)

[arXiv](https://arxiv.org/pdf/2502.02205) | [Paper](https://openreview.net/forum?id=XGJ33p4qwt)

Official repo for the paper [From Uncertain to Safe: Conformal Adaptation of Diffusion Models for Safe PDE Control](https://arxiv.org/pdf/2502.02205)<br />
[Peiyan Hu](https://peiyannn.github.io/)\*, [Xiaowei Qian](https://xweiq.github.io/)\*, [Wenhao Deng](https://w3nhao.github.io/), [Rui Wang](https://openreview.net/profile?id=~Rui_Wang56), [Haodong Feng](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en), [Ruiqi Feng](https://weenming.github.io/), [Tao Zhang](https://zhangtao167.github.io/), [Long Wei](https://longweizju.github.io/), [Yue Wang](https://scholar.google.com/citations?hl=zh-CN&user=fGv5irIAAAAJ), [Zhi-Ming Ma](http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html), [Tailin Wu](https://tailin.org/)<br />
**ICML** 2025.

We propose safe diffusion models for PDE Control, which introduces the uncertainty quantile as model uncertainty quantification to achieve optimal control under safety constraints through both post-training and inference phases.

<a href="url"><img src="img/overview.pdf" align="center" width="700" ></a>

## Environment

Run the following commands to install dependencies. In particular, when running the 2D control task, the Python version must be 3.8 due to the requirement of the Phiflow software.

```code
bash env.sh
```

## Dataset
The dataset files can be downloaded via [this link](http://drive.google.com/drive/u/0/folders/1abh34Ottw5K2qT-BTWQ-t18acCtgnRZL?dmr=1&ec=wgc-drive-hero-goto).

Please place the 1D Burgers' dataset in the `1D/datasets/free_u_f_1e5/` folder, place the 2D Smoke dataset in the `2d/data` folder, and place the Tokamak dataset on the `tokamak/tokamak_dataset` folder.

## Checkpoints
The checkpoints can be downloaded via [this link](http://drive.google.com/drive/u/0/folders/1abh34Ottw5K2qT-BTWQ-t18acCtgnRZL?dmr=1&ec=wgc-drive-hero-goto).

Please place the 1D Burgers' checkpoint in the `1D/experiments/checkpoints/turbo-1` folder, place the 2D Smoke checkpoint in the `2d/results` folder, and place the Tokamak checkpoint on the `tokamak/experiments` folder.

## Experiments
### 1D Burgers' Equation
```code
bash /1D/pretrain_eval.sh && bash /1D/reproduce_InfFT.sh
```

### 2D Smoke
```code
bash /2d/scripts/train.sh && bash /2d/scripts/posttrain.sh && bash /2d/scripts/finetune.sh
```

### Tokamak
```code
python /tokamak/pretrain.py && bash /tokamak/scripts/posttrain.sh && bash /tokamak/scripts/finetune.sh
```

## Related Projects
* [WDNO](https://github.com/AI4Science-WestlakeU/wdno) (ICLR 2025): We introduce Wavelet Diffusion Neural Operator (WDNO), a novel method for generative PDE simulation and control, to address diffusion models' challenges of modeling system states with abrupt changes and generalizing to higher resolutions.

* [CL-DiffPhyCon](https://github.com/AI4Science-WestlakeU/CL_DiffPhyCon) (ICLR 2025): We introduce an improved, closed-loop version of DiffPhyCon. It has an asynchronous denoising schedule for physical systems control tasks and achieves closed-loop control with significant speedup of sampling efficiency.

* [DiffPhyCon](https://github.com/AI4Science-WestlakeU/diffphycon) (NeurIPS 2024): We introduce DiffPhyCon which uses diffusion generative models to jointly model control and simulation of complex physical systems as a single task. 

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
  hu2025from,
  title={From Uncertain to Safe: Conformal Adaptation of Diffusion Models for Safe {PDE} Control},
  author={Peiyan Hu and Xiaowei Qian and Wenhao Deng and Rui Wang and Haodong Feng and Ruiqi Feng and Tao Zhang and Long Wei and Yue Wang and Zhi-Ming Ma and Tailin Wu},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=XGJ33p4qwt}
}
```