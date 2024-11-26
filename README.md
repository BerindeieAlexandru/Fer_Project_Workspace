# Ensemble Models For Facial Emotion Recognition

**Abstract:**

Facial Expression Recognition (FER) is a significant machine learning challenge, focused on the accurate identification of human facial expressions. Despite recent advancements, enhancing performance and obtaining reliable results in real-world conditions continues to pose substantial difficulties. In this repo, will be presented several FER approaches, focusing on FER2013 and FER+ datasets, a widely used for this task. In the provided paper we present the problem that we are trying to solve then we present the theoretical background talking about related work, the notions used and about the dataset. We then present the methodology of research that we used to select the papers and the methods used. We present five popular and effective approaches focusing on their contribution and the results, some of those being ResEmoteNet, PAtt-Lite, Residual Masking Network. We implement them and try to improve the presented results using different augmentations on the dataset, changing hyper-parameters, employing an ensemble approach (composed of the best performing models) and even fine-tuning the models. Finally, we evaluate our method by comparing the results with leading models in the field, demonstrating a significant increase in accuracy. The results underscore the effectiveness of our approach, offering a comprehensive solution to the persistent challenges in FER, and paving the way for more empathetic AI, advanced human computer interfaces, and improved mental health support systems.

## Table of Contents
- [Ensemble Models For Facial Emotion Recognition](#ensemble-models-for-facial-emotion-recognition)
  - [Table of Contents](#table-of-contents)
  - [Latest Changes](#latest-changes)
  - [Benchmark Results](#benchmark-results)
  - [ResEmoteNet](#resemotenet)
  - [PAtt-Lite](#patt-lite)
  - [Residual Masking Network](#residual-masking-network)
  - [Other Mentions](#other-mentions)

## Latest Changes

*   **2024-11-08:** Initial release with ResEmoteNet
*   **2024-11-22:** Added final version of ResEmoteNet (see comits)

## Benchmark Results

| Model                    | Dataset | Ensemble | Accuracy (%) | Finetune |
| ------------------------ | :----------: | :----------: | :-------: | :----------: |
| ResEmoteNet              |     XX.X     |     XX.X     |    XX.X   |     XX.X     |
| PAtt-Lite                |     XX.X     |     XX.X     |    XX.X   |     XX.X     |
| Residual Masking Network |     XX.X     |     XX.X     |    XX.X   |     XX.X     |

## ResEmoteNet

ResEmoteNet is a deep residual network architecture specifically designed for facial emotion recognition. It leverages residual connections to improve gradient flow and enable the training of deeper networks. The architecture consists of several residual blocks, each containing convolutional layers, batch normalization, and ReLU activation functions. 

**Key Features:**

*   Deep residual architecture for improved performance.
*   Utilizes convolutional layers for feature extraction.
*   Employs batch normalization for faster convergence.

**Paper Reference:**

*   [Cite the paper here](link-to-paper)

## PAtt-Lite

PAtt-Lite is a lightweight attention-based model for facial emotion recognition. It incorporates a pyramidal attention mechanism to capture both local and global features in facial expressions. The model is designed to be computationally efficient, making it suitable for real-time applications.

**Key Features:**

*   Pyramidal attention mechanism for multi-scale feature extraction.
*   Lightweight architecture for efficient computation.
*   Suitable for real-time emotion recognition.

**Paper Reference:**

*   [Cite the paper here](link-to-paper)

## Residual Masking Network

The Residual Masking Network introduces a novel approach to facial emotion recognition by using a masking mechanism to focus on the most discriminative facial regions. The network learns to generate a residual mask that highlights important features while suppressing irrelevant information. This allows the model to effectively handle variations in pose, illumination, and occlusion.

**Key Features:**

*   Residual masking mechanism for feature selection.
*   Robustness to pose, illumination, and occlusion variations.
*   Focuses on the most discriminative facial regions.

**Paper Reference:**

*   [Cite the paper here](link-to-paper)

## Other Mentions

 **Future Work:** Plans for future work include exploring other ensemble methods and incorporating additional datasets.
 **Related Resources:**
* [Kaggle link for extra experiments](https://www.kaggle.com/code/alexandruberindeie/fer-workspace)