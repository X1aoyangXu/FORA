# Feature-Oriented Reconstruction Attack
**Abstract:** Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.

**Our paper has been accepted at CVPR 2024!** 🎉🎉

**You can find the paper here!** 👉 https://arxiv.org/abs/2405.04115

# Code
We provide a demo of FORA on CIFAR-10 dataset in this repository, and the implementation is based on Pytorch. We provide a script to run the code. Before running script, read scripts carefully and set parameters as you want.

`sh run.sh`

The directory contains the following files to run FORA:
* `main.py`:  This file contains the main implementation of FORA.
* `model.py`: This file contains all the models we used for CIFAR-10.
* `splitnn.py`: This file contains the code to run a split learning task.
* `utils.py`: This file contains various functions to run main file.

# Cite our work:
```
@inproceedings{xu2024stealthy,
  title={A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning},
  author={Xu, Xiaoyang and Yang, Mengda and Yi, Wenzhe and Li, Ziang and Wang, Juan and Hu, Hongxin and Zhuang, Yong and Liu, Yaxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12130--12139},
  year={2024}
}
```
# Acknowledgment:
Our partial code is based on the following contributors.

Split learning framework: https://github.com/Koukyosyumei/Attack_SplitNN.

Multiple Kernel Maximum Mean Discrepancy: https://github.com/thuml/Transfer-Learning-Library.

