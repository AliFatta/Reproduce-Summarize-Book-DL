# TensorFlow in Action 🚀

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Complete-green)

**Reference Book:** *TensorFlow in Action* by Thushan Ganegedara  
**Repository Goal:** To reproduce, explain, and expand upon the concepts presented in the book, providing a hands-on guide to modern Deep Learning and MLOps using TensorFlow 2.x.

---

## 📖 Repository Overview

### Purpose
This repository is a comprehensive companion to Thushan Ganegedara's *TensorFlow in Action*. It bridges the gap between theoretical machine learning concepts and production-ready code. Each notebook corresponds to a chapter in the book, containing detailed theoretical explanations, mathematical intuition, and runnable code examples.

### Learning Goals
By working through this repository, you will:
1.  **Master TensorFlow 2:** Transition from basic tensors to complex custom training loops and Keras layers.
2.  **Build Core Architectures:** Implement CNNs (Inception, ResNet), RNNs (LSTM, GRU), and Transformers from scratch.
3.  **Understand the "Why":** Gain intuition on *why* certain architectures work (e.g., Attention mechanisms, Residual connections).
4.  **Deploy Models:** Learn end-to-end MLOps using TFX, Docker, and TensorFlow Serving.

---

## 📚 Chapter-by-Chapter Summary

### Part 1: Foundations of TensorFlow 2 and Deep Learning

| Chapter | Title | Summary |
| :--- | :--- | :--- |
| **01** | [The Amazing World of TensorFlow](Chapter_01_The_amazing_world_of_TensorFlow.ipynb) | Introduction to the ecosystem. Benchmarking **CPU vs. GPU** performance for matrix operations to understand hardware acceleration. |
| **02** | [TensorFlow 2](Chapter_02_TensorFlow_2.ipynb) | Deep dive into atomic building blocks: `tf.Tensor`, `tf.Variable`, and `tf.Operation`. Exploring **Eager Execution** vs. **Graph Execution** (`@tf.function`). |
| **03** | [Keras and Data Retrieval](Chapter_03_Keras_and_data_retrieval_in_TensorFlow_2.ipynb) | Mastering the Keras APIs (**Sequential, Functional, Subclassing**) and building high-performance data pipelines with `tf.data`. |
| **04** | [Dipping Toes in Deep Learning](Chapter_04_Dipping_toes_in_deep_learning.ipynb) | Building the "Big Three": **Autoencoders** (Dense), **Image Classifiers** (CNN), and **Forecasters** (RNN) from scratch. |
| **05** | [State-of-the-Art: Transformers](Chapter_05_State_of_the_art_in_deep_learning_Transformers.ipynb) | Implementing the **Transformer** architecture (Encoder/Decoder) and **Self-Attention** mechanisms from the ground up without using pretrained libraries. |

### Part 2: Computer Vision - Looking at the World

| Chapter | Title | Summary |
| :--- | :--- | :--- |
| **06** | [Image Classification with CNNs](Chapter_06_Teaching_machines_to_see_Image_classification_with_CNNs.ipynb) | Building complex CNN architectures like **InceptionNet** (GoogLeNet) to classify images in the **Tiny ImageNet** dataset. |
| **07** | [Teaching Machines to See Better](Chapter_07_Teaching_machines_to_see_better.ipynb) | Advanced techniques: **Transfer Learning**, Residual Connections, and **Grad-CAM** for model interpretability/visualization. |
| **08** | [Image Segmentation](Chapter_08_Telling_things_apart_Image_segmentation.ipynb) | Pixel-level classification using the **U-Net** architecture. Implementing Transposed Convolutions and Skip Connections. |

### Part 3: Natural Language Processing - Understanding Text

| Chapter | Title | Summary |
| :--- | :--- | :--- |
| **09** | [NLP: Sentiment Analysis](Chapter_09_NLP_with_TensorFlow_Sentiment_Analysis.ipynb) | Processing text data with **LSTMs**. Topics include tokenization, embedding layers, and handling class imbalance. |
| **10** | [NLP: Language Modeling](Chapter_10_Natural_language_processing_with_TensorFlow_Language_modeling.ipynb) | Generative AI foundations. Building a **GRU**-based model to generate text and measuring performance with **Perplexity**. |
| **11** | [Seq2Seq Learning: Part 1](Chapter_11_Sequence_to_sequence_learning_Part_1.ipynb) | Building a **Machine Translation** system (English to German) using an Encoder-Decoder RNN architecture and Teacher Forcing. |
| **12** | [Seq2Seq Learning: Part 2](Chapter_12_Sequence_to_sequence_learning_Part_2.ipynb) | Enhancing translation with **Bahdanau Attention**. Implementing custom Keras layers to visualize alignment heatmaps. |
| **13** | [Transformers in Practice](Chapter_13_Transformers.ipynb) | Using **BERT** and **Hugging Face** for downstream tasks like Spam Classification and Question Answering (SQuAD). |

### Part 4: Production and MLOps

| Chapter | Title | Summary |
| :--- | :--- | :--- |
| **14** | [TensorBoard](Chapter_14_TensorBoard_Big_brother_of_TensorFlow.ipynb) | Visualizing training metrics, debugging weights with histograms, and projecting high-dimensional embeddings. |
| **15** | [TFX and MLOps](Chapter_15_TFX_MLOps_and_deploying_models_with_TensorFlow.ipynb) | Building an end-to-end production pipeline with **TensorFlow Extended (TFX)**. Data validation, transformation, training, and deployment with **Docker**. |

### Appendices

| Appx | Title | Summary |
| :--- | :--- | :--- |
| **A** | [Environment Setup](Appendix_A_Environment_Setup.ipynb) | Step-by-step guide to installing TensorFlow, GPU drivers, and library dependencies. |
| **B** | [CV Deep Dive](Appendix_B_Computer_Vision.ipynb) | Technical implementation of **Grad-CAM** for nested/complex models (like InceptionResNetV2). |
| **C** | [NLP Deep Dive](Appendix_C_NLP_Attention_Deep_Dive.ipynb) | Visualizing **Positional Encodings** and **Attention Masks** (Look-Ahead Mask) to understand Transformer geometry. |

---

## 🛠️ Tools & Technologies

* **Core Framework:** TensorFlow 2.x, Keras
* **Data Processing:** NumPy, Pandas, TensorFlow Datasets (TFDS)
* **NLP:** NLTK, Hugging Face Transformers, TensorFlow Text
* **Deployment:** TFX, Docker, TensorFlow Serving
* **Visualization:** Matplotlib, TensorBoard

## 🧠 Learning Outcome

After completing this repository, you will have transitioned from a student of Machine Learning to a practitioner capable of:
1.  **Architecting** custom neural networks for Vision and NLP tasks.
2.  **Debugging** models using professional visualization tools.
3.  **Deploying** scalable ML pipelines that handle data validation and model serving in production environments.

---

## 🤝 Credits
All theoretical concepts and base code structures are derived from *TensorFlow in Action* by Thushan Ganegedara. This repository adapts and expands on those examples for educational purposes.
