<div align="center">

<!-- ใส่ภาพแบนเนอร์ด้านบนสุด (เปลี่ยน URL เป็นภาพโปรเจกต์ของคุณได้เลย) -->
![TB Cough Analysis Banner](https://placehold.co/1200x300/111111/4CAF50?text=COUGH-ANALYSIS+FOR+TB&font=Press+Start+2P)

# Cough Analysis for Tuberculosis Screening 🫁

<!-- ป้าย Badges (ใช้สไตล์ for-the-badge) -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/STATUS-R%26D-FFD700?style=for-the-badge&labelColor=555555" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/COURSE-PRACTICAL%20DATA%20SCIENCE-5865F2?style=for-the-badge&labelColor=555555" alt="Course"></a>
  <a href="#"><img src="https://img.shields.io/badge/PYTHON-3.9+-3776AB?style=for-the-badge&labelColor=555555&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TASK-AUDIO%20CLASSIFICATION-E06666?style=for-the-badge&labelColor=555555" alt="Task"></a>
</p>

</div>

**Research Topic:** *Cross-Dataset Generalization of Tuberculosis Screening from Cough Audio to Thai Smartphone-based Clinical Recordings*

---

## 📖 Abstract

While AI-based cough analysis is a promising non-invasive tool for Tuberculosis (TB) screening, generalizing models from public datasets to local clinical settings remains challenging due to domain shifts. This study evaluates cross-dataset generalization by training models on the international CODA dataset and testing them on a target domain of 512 smartphone-recorded cough events from Siriraj Hospital, Thailand. We analyzed various acoustic features, classifiers, source-domain selections, and the integration of clinical variables. 

Results showed that the **HeAR embedding combined with a Multilayer Perceptron (MLP)** achieved the best target-domain performance. Notably, training on regionally selected CODA subsets outperformed using the full dataset, and incorporating clinical features further enhanced model robustness. Explainability analysis also provided insights into the cough regions driving predictions. These findings demonstrate the feasibility of leveraging public datasets for TB screening, while highlighting the critical need to address domain shift, data selection, and interpretability prior to local deployment.

---

## 🎯 Objectives

| | |
| :--- | :--- |
| 📚 **Curriculum** | To implement and evaluate machine learning pipelines as part of the Practical Data Science curriculum. |
| 🔍 **Domain Shift** | To investigate the domain shift between public cough audio datasets and Thai smartphone-based recordings. |
| 🛠️ **Feature Extraction** | To develop robust audio processing and feature extraction techniques (e.g., using the first 0.5 seconds of cough signals). |
| 🤖 **Benchmarking** | To benchmark various deep learning architectural paradigms for this specific audio classification task. |
