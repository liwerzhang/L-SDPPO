
```markdown
# L-SDPPO: Policy Optimization of Spiking Diffusion Models for  Intra-vehicular Robotic Manipulation
### 🎬 Supplementary Video
The experimental results and comparisons are available in the [Supplementary Video](./assets/demo_video.mp4).

[![Double-Blind](https://img.shields.io/badge/Review-Double_Blind-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This repository contains the official anonymous code for the paper "L-SPPO: A Proficient and Energy-Economical Spiking Diffusion Policy for Microgravity Manipulation" currently under double-blind review for IEEE Robotics and Automation Letters (RA-L).

## 🚀 Overview
Intra-vehicular robots (IVRs) are essential for assisting astronauts in microgravity environments. We introduce **L-SDPPO**, a spiking diffusion policy framework that achieves a **97% success rate** with only **36.45% energy consumption** of traditional models.

### 🤖 Representative Tasks (Task I - V)

| Task I: Hatch Opening | Task III: Precision Capping | Task V: Long-horizon Stowing |
| :---: | :---: | :---: |
| ![Hatch Opening](./assets/door.gif) | ![Precision Capping](./assets/onlypicklid.gif) | ![Long-horizon Stowing](./assets/drawer.gif) |
| *Rotating handle to unlock* | *Precise spatial alignment* | *Multi-step planning* |

> **Note**: Our method consistently outperforms traditional DP and behavior cloning across all tasks, especially in handling unconstrained drift in microgravity.
## 🛠️ Installation 

1. Clone the repository:
```console
git clone <your_anonymous_github_url>
cd L-SDPPO-Code
```

2. Create and activate a conda environment:
```console
conda create -n lsdppo python=3.8 -y
conda activate lsdppo
pip install -e .
```

3. Install environment dependencies for microgravity simulations (compatible with Robomimic and customized Robosuite environments):
```console
pip install -e .[robomimic]
```

4. Set environment variables for data and logging directories (default is `data/` and `log/`):
```console
source script/set_path.sh
```

## 🎮 Evaluated Microgravity Tasks

[cite_start]The framework is rigorously evaluated across five distinct space cabin simulation scenarios: [cite: 65, 180]

* [cite_start]**Task I:** Constrained Hatch Actuation [cite: 288]
* [cite_start]**Task II:** Dynamic Tumbling Target Stowing (Box) [cite: 290]
* [cite_start]**Task III:** Precision Container Capping (Box Closing) [cite: 290]
* [cite_start]**Task IV:** Sequential Panel Operation (Workbench Operation) [cite: 292]
* [cite_start]**Task V:** Long-horizon Drawer-based Stowing [cite: 293, 295]

## 💻 Usage - Pre-training

[cite_start]To pre-train the Spiking Diffusion Policy on the expert dataset via Behavior Cloning (MSE Loss): [cite: 180]

```bash
# Example: Task V (Drawer-based Stowing)
python script/run.py --config-name=pre_snn_diffusion \
    --config-dir=cfg/robomimic/pretrain/drawer
```

## 📈 Usage - Fine-tuning (RL with PPO)

[cite_start]To fine-tune the pre-trained SNN policy using PPO and the SDLI mechanism: [cite: 181, 243]

```bash
# Example: Task III (Box Closing)
python script/run.py --config-name=ft_ppo_diffusion_snn \
    --config-dir=cfg/robomimic/finetune/drawer \
    base_policy_path=/path/to/pretrained/checkpoint/state_x.pt
```

## 📊 Evaluation

[cite_start]To evaluate a trained policy (either pre-trained or fine-tuned) and compute the Success Rate (SR) and Average Reward (AR): [cite: 280, 281]

```bash
python script/run.py \
    --config-name=eval_diffusion_snn \
    --config-dir=cfg/robomimic/eval/drawer \
    base_policy_path=/path/to/pretrained/checkpoint/state_x.pt \
    ft_denoising_steps=10 \
    env.save_video=True \
    render_num=1
```

## 📄 License

This repository is released under the MIT license. See `LICENSE`.

## 🙏 Acknowledgement

Our implementation builds upon several excellent open-source repositories:

* **Diffuser** for general diffusion processes.
* **Diffusion Policy** for visuomotor policy wrappers.
* **DPPO** for PPO integration with diffusion models.
* **Robomimic** for benchmark environments.
