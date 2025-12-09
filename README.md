

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


## 🎯 **Usage Examples**

### **Run Single / Multi-Robot Diffusion Planning**

```
python inference_multi.py --model_id EnvSpheres3D-RobotPanda --seed 0 --n_samples 50 --n_robots 5 --robot_spacing 0.3 --render True
```

### **Test the Transformer Policy in MultiArm Environment**

```
python main.py --mode benchmark --tasks_path benchmark/ --load ours/ours.pth --num_processes 1 --gui
```


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

