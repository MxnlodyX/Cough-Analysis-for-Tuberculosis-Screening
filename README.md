<div align="center">

<!-- ใส่ภาพแบนเนอร์ด้านบนสุด -->
![TB Cough Analysis Banner](https://placehold.co/1200x300/111111/4CAF50?text=COUGH-ANALYSIS+FOR+TB&font=Press+Start+2P)

# Cough Analysis for Tuberculosis Screening 🫁

<!-- ป้าย Badges (ใช้สไตล์ for-the-badge) -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/STATUS-ONGOING%20RESEARCH-007BFF?style=for-the-badge&labelColor=555555" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/COURSE-PRACTICAL%20DATA%20SCIENCE-5865F2?style=for-the-badge&labelColor=555555" alt="Course"></a>
  <a href="#"><img src="https://img.shields.io/badge/PYTHON-3.9+-3776AB?style=for-the-badge&labelColor=555555&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TASK-AUDIO%20CLASSIFICATION-E06666?style=for-the-badge&labelColor=555555" alt="Task"></a>
</p>

</div>

**Research Topic:** *Cross-Dataset Generalization of Tuberculosis Screening from Cough Audio to Thai Smartphone-based Clinical Recordings*

This project is divided into two main phases, combining coursework experiments with in-depth clinical research:

---
## 🔬 Part 1: Multi-Features Evaluation (Practical Data Science)

The primary objective of this phase is to evaluate and identify the most effective acoustic features for predicting Pulmonary Tuberculosis (PTB). Through rigorous experimentation with various feature extraction methods, our findings indicate that **HeAR Embeddings** provide the best overall performance and most robust representation when fed into our classification models.

---

## 🔬 Part 2: Cross-Dataset Domain Generalization (Research)

Building upon the optimal features identified in Part 1, this phase focuses on the challenges of deploying AI models in real-world clinical settings. 

While AI-based cough analysis is a promising non-invasive tool for TB screening, generalizing models from public datasets to local clinical settings remains challenging due to domain shifts. This study evaluates cross-dataset generalization by training models on the international CODA dataset and testing them on a target domain of **539 smartphone-recorded cough events from Siriraj Hospital, Thailand**. We analyzed various acoustic features, classifiers, source-domain selections, and the integration of clinical variables. 

**Key Findings:**
*   🏆 **Best Model:** The **HeAR embedding combined with a Multilayer Perceptron (MLP)** achieved the strongest target-domain performance.
*   📊 **Data Selection:** Training on regionally selected CODA subsets outperformed using the full dataset, suggesting that source-domain composition strongly influences transfer under distribution shift.
*   🏥 **Clinical Integration:** Incorporating clinical features further enhanced model robustness in cross-dataset settings.
*   🧠 **Interpretability:** Explainability analysis provided preliminary insights into the specific cough regions driving model predictions.

These findings demonstrate the feasibility of leveraging public datasets for TB screening, while highlighting the critical need to address domain shift, data selection, and interpretability prior to local deployment.
