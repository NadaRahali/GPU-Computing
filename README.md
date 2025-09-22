# GPU Computing: kNN and MLP Acceleration  

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/GPU_computing_logo.png" alt="GPU Logo" width="200"/>
</p>

## Overview  
This project investigates the impact of **GPU acceleration** on machine learning algorithms.  
We focused on:  
- **k-Nearest Neighbors (kNN)**  
- **Multilayer Perceptron (MLP)**  

Both models were implemented and tested on **CPU vs GPU** to measure runtime differences. The work was carried out as part of the **GPU Computing course at LUT University**.  

---

## Objectives  
- Implement kNN and MLP for classification.  
- Compare runtime and accuracy between CPU and GPU execution.  
- Evaluate how dataset size and algorithm type affect GPU speedup.  

---

## Dataset  
- **File:** `MLoGPU_data3_train.csv` – dataset provided by the course.  
- **Samples:** 4,000  
- **Features:** 7 numerical values per sample  
- **Classes:** 7 (multi-class classification)  
- **Preprocessing:**  
  - Min–Max normalization  
  - Labels cast to integers  
  - Train/test split (80/20, stratified)  

---

## Technologies Used  
- **Python 3.10**  
- **Google Colab (GPU runtime)**  
- **CuPy** – GPU array operations & custom CUDA kernels  
- **PyTorch** – neural network implementation (MLP)  
- **NumPy** – CPU-based array operations  
- **Matplotlib** – visualizations  
- **Scikit-learn** – preprocessing (train/test split, scaling)

---

## Results  

### k-Nearest Neighbors (kNN)  
- **Best accuracy:** ~52% (k = 1)  
- **Runtime comparison:**  
  - CPU avg: ~0.50s  
  - GPU avg: ~0.006s  
- **Speedup:** Up to **170x** for small k, ~30–40x for larger k  
- **Observation:** GPU acceleration was very effective for distance calculations, but accuracy was limited by class imbalance.  

### Multilayer Perceptron (MLP)  
- **Architecture:** 7 → 64 → 64 → 7 (ReLU + CrossEntropyLoss)  
- **Training setup:** 100 epochs, Adam optimizer (lr=0.001)  
- **Accuracy:**  
  - CPU: 52.25%  
  - GPU: 53.87%  
- **Runtime:**  
  - Training: CPU ~9.95s, GPU ~10.38s  
  - Inference: CPU ~0.01s, GPU ~0.012s  
- **Observation:** GPU overhead outweighed benefits for this small dataset and simple model.  

---

## Key Findings  
- GPU acceleration provides **significant speedups** for highly parallelizable methods like kNN.  
- For small networks (MLP) and limited data, GPU benefits are minimal.  
- Dataset imbalance and overlapping features limited accuracy more than compute power.  
- Writing a **custom CUDA kernel** highlighted the importance of memory management and thread-level parallelism.  

---

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/GPU-Computing.git
   cd GPU-Computing
   ```

---

## Authors  
This project was completed as part of **GPU Computing (LUT University)** by:  
- **Nada Rahali** – MLP implementation, report writing  
- **Tanjuma Haque** – kNN implementation (CPU + GPU kernel), documentation  

---

## License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

 

