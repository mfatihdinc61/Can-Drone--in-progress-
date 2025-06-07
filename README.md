# Can-Drone--in-progress-
 Utilizing VLA Drones for Some Tasks and Determining the Performance Metrics for Future Projects

# 🛸 Can: Vision-Language Integrated Smart Drone Project

**Can** is an experimental drone project that aims to integrate **Vision-Language Pre-training (VLP)** systems with a **small-scale autonomous drone**. The long-term goal is to create a general-purpose smart drone that can be controlled through natural language and capable of performing complex real-world tasks based on environmental understanding.

---

## 🎯 Project Goals

1. 📏 **Establish Evaluation Metrics**  
   Develop standardized **performance metrics** for evaluating and benchmarking intelligent drone systems.

2. 🔥 **Applied Smart Drone Use-Cases**
   - **Forest fire inspection & early extinguishing** (detecting and reacting to fires in initial stages)
   - **Area guarding / autonomous surveillance**
   - **Smart delivery systems** that can adapt to dynamic environments

---

## 🧠 Vision-Language Integration

The project is built upon modern **Vision-Language Pre-training (VLP)** techniques, aiming to empower the drone with the ability to:
- Interpret natural language commands
- Perceive its environment through onboard cameras
- Make context-aware decisions autonomously

---

## 🎬 Demonstrations & Code

Below are three key prototypes that demonstrate different aspects of the project:

### 1. Prompt-Controlled Real Drone

> **Hardware**: DJI Tello  
> **Language Model**: OpenAI GPT via API

📹 **Video**: [Watch](./videos/prompt_controlled_drone.mp4)  
💻 **Script**: [`prompt_controlled_drone.py`](./code/telloDrone5.py)

**Description**:  
A real-world implementation where a Tello drone is controlled via natural language prompts such as:
- `"Take off"`
- `"Move forward and rotate"`
- `"Land if battery is below threshold"`

---

### 2. Microsoft's AirSim-GPT Integration

> Based on Microsoft’s open-source [PromptCraft-Robotics](https://github.com/microsoft/PromptCraft-Robotics/tree/main/chatgpt_airsim)

📹 **Video**: [Watch](./videos/airsim_gpt_demo.mp4)  

**Description**:  
We tested the integration of GPT with Microsoft's **AirSim** drone simulation environment. This forms a testbed for simulating and refining prompt-based drone behavior before real-world deployment.

---

### 3. BLIP (VLP) Integration Test

📹 **Video**: [Watch](./videos/blip_test.mp4)  
💻 **Script**: [`blip_droidcam.py`](./code/blip_droidcam.py)

**Description**:  
BLIP (Bootstrapped Language-Image Pretraining) was tested to evaluate its capability to understand drone surroundings via images. This is a **critical module** planned to be embedded in the drone for contextual awareness and decision-making.

---

## 🛠️ Technologies Used

- **Python**
- **OpenAI GPT API**
- **BLIP (Salesforce Vision-Language Model)**
- **AirSim (Microsoft)**
- **DJI Tello SDK**

---

## 🧪 Future Plans

- Integrate BLIP and GPT for full onboard reasoning
- Develop drone-specific datasets for VLP finetuning
- Establish and publish standardized performance evaluation framework
- Expand drone hardware capabilities (e.g., custom fire extinguishing payload)
