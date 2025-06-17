# Lunar Lander Landing with Decision Transformer

## Project Overview

This project implements an **Optimized Decision Transformer (DT)** model to tackle the classic Lunar Lander environment from Gymnasium. The Decision Transformer reimagines reinforcement learning as a sequence modeling problem, using a transformer architecture to predict actions conditioned on desired future returns, states, and past actions. This approach allows the agent to learn goal-conditioned policies directly from expert trajectories.

The primary objective is to train an autonomous agent to safely and efficiently land the Lunar Lander module on a designated landing pad, demonstrating precise control and robust decision-making.

## Key Features

* **Developed and implemented an Optimized Decision Transformer (DT) model in PyTorch** to solve the Lunar Lander reinforcement learning environment, showcasing expertise in advanced sequence modeling for control.
* **Engineered a robust training pipeline** incorporating mixed precision training, gradient clipping, early stopping, and learning rate scheduling, leading to efficient and stable model convergence.
* **Processed and leveraged a dataset of 5,000 expert trajectories**, employing sophisticated filtering and data augmentation techniques to maximize training effectiveness and model robustness.
* **Achieved an outstanding 99% success rate** in landing the Lunar Lander, with an **average return of 283.76** (± 27.64) across 100 evaluation episodes, significantly surpassing the success threshold of 200.
* **Designed and integrated a real-time Pygame visualization module** for the Lunar Lander environment, enabling dynamic observation and analysis of the trained agent's performance.

## Results

After training on a curated set of expert trajectories, the Decision Transformer model demonstrated highly successful performance in the Lunar Lander environment:

* **Episodes evaluated:** 100
* **Mean Return:** 283.76 ± 27.64
* **Min Return:** 69.63
* **Max Return:** 322.04
* **Success Rate (Return >= 200):** 99.0%

These results highlight the model's ability to learn and generalize optimal landing strategies, consistently achieving high scores and successful landings.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone (https://github.com/SyedNaveedM/Lunar_Lander_DT)
    cd Lunar_Lander_DT
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install torch numpy gymnasium pygame tqdm
    ```
    * **Note:** Ensure you have the correct PyTorch version for your system, especially if using a GPU (CUDA). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more details.

4.  **Prepare Trajectories:**
    The project relies on a pre-generated dataset of expert trajectories. Ensure you have a file named `trajectories_5000.pkl` (or similar, as configured in `TRAJECTORY_FILE`) in a `datasets/` directory at the root of your project. If you don't have this file, you would typically need to generate expert trajectories using a pre-trained traditional RL agent (e.g., PPO, SAC). For this project, it's assumed this file exists or will be provided.

## Usage

### Training the Model

The main Jupyter Notebook `lunar_lander_train.ipynb` (or the equivalent Jupyter Notebook) handles the entire training process.
To train the Decision Transformer, simply run the cells in the notebook
