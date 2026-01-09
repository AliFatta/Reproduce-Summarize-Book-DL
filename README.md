# TensorFlow in Action

**Reference Book:** *TensorFlow in Action* by Thushan Ganegedara  
**Status:** In Progress 🚧

## 1. Repository Overview

### Purpose
This repository serves as a comprehensive, hands-on companion to Thushan Ganegedara's *TensorFlow in Action*. It aims to reproduce, explain, and summarize every chapter of the book to deepen theoretical understanding and practical skills in Machine Learning and Deep Learning using TensorFlow 2.

### Learning Goals
- **Master TensorFlow 2:** From basic tensors to complex production pipelines.
- **Implement Core Algorithms:** Build CNNs, RNNs, and Transformers from scratch.
- **Theory to Practice:** Bridge the gap between mathematical concepts and working code.
- **Production Skills:** Learn MLOps, model deployment, and monitoring with TensorBoard and TFX.

## 2. Chapter-by-Chapter Summary

### Part 1: Foundations of TensorFlow 2 and Deep Learning

* **Chapter 1: The amazing world of TensorFlow**
    * **Summary:** An introduction to the TensorFlow ecosystem, distinguishing between CPUs, GPUs, and TPUs. Covers the "when" and "why" of using TensorFlow versus other tools.
    * **Skills:** Understanding hardware acceleration, defining the scope of TensorFlow projects.

* **Chapter 2: TensorFlow 2**
    * **Summary:** Deep dive into the core building blocks: `tf.Variable`, `tf.Tensor`, and `tf.Operation`. Explores how TensorFlow executes code under the hood (Eager Execution).
    * **Skills:** Low-level TensorFlow manipulation, understanding computational graphs.

* **Chapter 3: Keras and data retrieval in TensorFlow 2**
    * **Summary:** Introduction to the high-level Keras API (Sequential, Functional, Subclassing) and efficient data loading pipelines using `tf.data`.
    * **Skills:** Building model architectures, creating scalable data input pipelines.

* **Chapter 4: Dipping toes in deep learning**
    * **Summary:** Implementing fundamental neural networks: Fully Connected (Dense), Convolutional (CNN), and Recurrent (RNN).
    * **Skills:** Building basic classifiers and regressors, understanding layer types.

* **Chapter 5: State-of-the-art in deep learning: Transformers**
    * **Summary:** A theoretical and practical breakdown of the Transformer architecture, including self-attention mechanisms.
    * **Skills:** Implementing Transformers, understanding Attention mechanisms.

### Part 2: Look Ma, No Hands! Deep Networks in the Real World

* **Chapter 6: Teaching machines to see: Image classification with CNNs**
    * **Summary:** Advanced image classification using complex CNN architectures like InceptionNet. Includes exploratory data analysis (EDA) for images.
    * **Skills:** Building state-of-the-art image classifiers, handling real-world image data.

* **Chapter 7: Teaching machines to see better: Improving CNNs and making them confess**
    * **Summary:** Techniques to reduce overfitting (dropout, augmentation) and interpret model decisions (Grad-CAM).
    * **Skills:** Regularization, model interpretability/explainability.

* **Chapter 8: Telling things apart: Image segmentation**
    * **Summary:** Tackling semantic segmentation tasks using architectures like DeepLabv3 and U-Net.
    * **Skills:** Pixel-level classification, advanced computer vision tasks.

* **Chapter 9: Natural language processing with TensorFlow: Sentiment analysis**
    * **Summary:** Processing text data for sentiment analysis using LSTM networks and word embeddings.
    * **Skills:** NLP preprocessing, sequence classification, working with text data.

* **Chapter 10: Natural language processing with TensorFlow: Language modeling**
    * **Summary:** Generating text using GRUs and Language Models. Covers decoding strategies like Beam Search.
    * **Skills:** Text generation, advanced sequence modeling.

### Part 3: Advanced Deep Networks for Complex Problems

* **Chapter 11: Sequence-to-sequence learning: Part 1**
    * **Summary:** Building an Encoder-Decoder architecture for machine translation (English to German).
    * **Skills:** Seq2Seq modeling, text translation pipelines.

* **Chapter 12: Sequence-to-sequence learning: Part 2**
    * **Summary:** Enhancing Seq2Seq models with the Bahdanau Attention mechanism.
    * **Skills:** Implementing custom Attention layers, improving translation quality.

* **Chapter 13: Transformers**
    * **Summary:** Advanced applications of Transformers using BERT and Hugging Face libraries for tasks like Question Answering.
    * **Skills:** Transfer learning with BERT, using Hugging Face Transformers.

* **Chapter 14: TensorBoard: Big brother of TensorFlow**
    * **Summary:** Using TensorBoard for visualization, metric tracking, and profiling model performance.
    * **Skills:** Model monitoring, debugging, performance profiling.

* **Chapter 15: TFX: MLOps and deploying models with TensorFlow**
    * **Summary:** Building end-to-end MLOps pipelines using TensorFlow Extended (TFX) for production deployment.
    * **Skills:** MLOps, model serving, creating production pipelines.

## 3. Tools & Technologies
* **Language:** Python 3.x
* **Core Library:** TensorFlow 2.x, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, TensorBoard
* **Environment:** Jupyter Notebook

## 4. Learning Outcome
By completing the notebooks in this repository, a learner will transition from a basic understanding of ML concepts to a proficient Deep Learning practitioner capable of:
1.  Architecting custom neural networks from scratch.
2.  Debugging and optimizing models for performance and accuracy.
3.  Deploying models into production environments using industry-standard MLOps practices.
