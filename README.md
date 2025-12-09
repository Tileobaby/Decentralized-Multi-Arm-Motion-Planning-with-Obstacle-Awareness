

# **Multi-Robot Motion Planning Extensions**

Extensions to **Motion Planning Diffusion (MPD)** and **Decentralized Multi-Arm Reinforcement Learning** for multi-robot trajectory generation and coordination.

This repository contains **only my own implementation and modifications**.
To run the full system, please first install the **original upstream projects** listed below.

---

## 🚀 **Overview**

This project provides two independent extensions:

### **1. Transformer Encoder for Multi-Arm Reinforcement Learning**

Replaces the LSTM-based neighbor encoder with a lightweight Transformer encoder:

* Supports variable numbers of neighboring robots
* Parallel attention instead of sequential LSTM
* Uses relative geometric encoding (Δpose, distances)
* Intended for decentralized multi-robot cooperation

> Note: Training a Transformer inside decentralized MARL proved highly unstable.
> Behavior Cloning converged, but RL performance remained poor.
> This implementation is provided for reference and reproducibility.

---

### **2. Multi-Robot Motion Planning Using Diffusion Models**

Extends **Motion Planning Diffusion (MPD)** from single-arm trajectories to **multi-robot joint planning**:

* Adds **inter-robot collision cost**
* Adds **Gibbs-style alternating denoising** for joint trajectory inference
* Naturally supports obstacle avoidance
* Scales to N robots (tested for N = 1–3)

This method does **not** require RL training and directly generates collision-aware full trajectories.

---

## 📦 **Repository Structure**

```
my-multiarm-extension/
│
├── transformer_encoder/
│   ├── transformer_state_encoder.py
│   └── README_notes.md
│
├── diffusion_multiarm/
│   ├── multiarm_cost.py
│   ├── gibbs_sampling.py
│   ├── run_multiarm_mpd.py
│   └── utils.py
│
├── patches/
│   ├── policy_patch.diff
│   └── config_patch.diff
│
└── examples/
    ├── example_config.yaml
    └── demo_script.py
```

---

## 🔧 **Installation**

Before using this repository, install the **two upstream frameworks**:

### **1. Motion Planning Diffusion (MPD)**

```
git clone https://github.com/eugenioc/motion-planning-diffusion.git
```

### **2. MultiArm Decentralized RL**

```
git clone https://github.com/columbia-ai-robotics/multiarm.git
```

Please ensure both projects can run independently.

---

## 🔗 **Integrating My Code with the Original Repositories**

### **A. Add the Transformer Encoder to the MultiArm RL Framework**

Copy the encoder into the policy module:

```
cp transformer_encoder/transformer_state_encoder.py  multiarm/policy/
```

(Optional) Apply patch:

```
patch -p1 < patches/policy_patch.diff
```

---

### **B. Add Multi-Robot Extensions to MPD**

```
cp diffusion_multiarm/*.py  motion-planning-diffusion/mpd/
```

Run the multi-robot diffusion planning:

```
python run_multiarm_mpd.py --num_robots 3
```

---

## 🎯 **Usage Examples**

### **Run Single / Multi-Robot Diffusion Planning**

```
python run_multiarm_mpd.py --num_robots 3 --with_obstacles 1
```

### **Test the Transformer Policy in MultiArm Environment**

```
python multiarm_env_test.py --policy transformer
```

### **Demo Script**

```
python examples/demo_script.py --robots 2
```

---

## 📊 **Features Added**

### ✔ Transformer for neighbor encoding

### ✔ Inter-robot collision cost

### ✔ Multi-robot Gibbs Sampling

### ✔ Configurable N-arm trajectory denoising

### ✔ Additional evaluation metrics

* Success rate
* Environment collision rate
* Inter-robot collision rate

### ✔ Architecture diagrams (included in paper, not in repo)

---

## 📚 **References**

This work builds on:

**Motion Planning Diffusion**
Carvalho et al., 2024
[https://github.com/eugenioc/motion-planning-diffusion](https://github.com/eugenioc/motion-planning-diffusion)

**Decentralized Multi-Arm Motion Planning (MultiArm)**
Wang et al., 2023
[https://github.com/columbia-ai-robotics/multiarm](https://github.com/columbia-ai-robotics/multiarm)

---

## 🙏 **Acknowledgments**

This project is part of my graduate research on robot motion planning.
The original authors’ work made this project possible.

---

## 💬 Questions?

