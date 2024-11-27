# Citation Context Classification

## Problem Statement

Citations are critical indicators of scholarly influence, reflecting the impact of research across various scientific disciplines. While citation counts have traditionally been used to assess the value of research, they lack context. Not all citations hold the same purpose or significance—some provide background information, while others critique or extend prior work. Accurate classification of citation context is essential to better understand academic discourse and evaluate the true influence of research papers.

The classification problem is divided into:

- **Subtask A**: Classify the purpose of citations into categories like `BACKGROUND`, `USES`, `COMPARES`, `MOTIVATION`, `EXTENSION`, and `FUTURE`.
- **Subtask B**: Assess the influence of citations by labeling them as `INCIDENTIAL` or `INFLUENTIAL`.

---

## Proposed Solution

To address these challenges, a combination of traditional machine learning and advanced deep learning techniques was utilized:

1. **Rule-Based Methods**: Basic logical structures to classify simpler citation contexts.
2. **Machine Learning Models**:
   - Random Forests
   - Support Vector Machines (SVM)
3. **Deep Learning Models**:
   - Fine-tuned **SciBERT**, specifically optimized for scientific text.
   - Enhanced SciBERT with **LSTM layers** to capture long-term dependencies in text.
4. **Weighted Loss Approach**:
   - Mitigates class imbalance in the dataset, ensuring underrepresented classes are not overlooked during training.

---

## Technologies Used

### Machine Learning Models:
- `scikit-learn`: For Random Forests and SVM.
- HuggingFace Transformers: Advanced transformer-based models.

### Deep Learning:
- **SciBERT**: A variant of BERT pretrained on scientific text.
- LSTM layers for sequence dependency.
- **Loss Function**: Weighted Cross-Entropy.
- **Optimizer**: Adam.

### Evaluation Metrics:
- **Accuracy**: Measure the overall correctness of predictions.
- **Macro F1 Score**: Evaluate performance across imbalanced datasets by balancing precision and recall.

---

This repository aims to improve the understanding and automation of citation context classification, leveraging both traditional and state-of-the-art approaches in machine learning and deep learning.
