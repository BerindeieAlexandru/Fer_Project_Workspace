# Ensemble Models For Facial Emotion Recognition

**Project description:**

Facial Expression Recognition (FER) is a significant machine learning challenge, focused on the accurate identification of human facial expressions. Despite recent advancements, enhancing performance and obtaining reliable results in real-world conditions continues to pose substantial difficulties.

In this REPOSITORY, we will present several FER approaches, focusing on FER2013 and FER+ datasets, a widely used for this task.

In the provided paper we present the problem that we are trying to solve then we present the theoretical background talking about related work, the notions used and about the dataset. We then present the methodology of research that we used to select the papers and the methods used. We present five popular and effective approaches focusing on their contribution and the results, some of those being ResEmoteNet, PAtt-Lite, Residual Masking Network.

We implement them and try to improve the presented results using different augmentations on the dataset, changing hyper-parameters, employing an ensemble approach (composed of the best performing models) and even fine-tuning the models. Finally, we evaluate our method by comparing the results with leading models in the field, demonstrating a significant increase in accuracy.

The results underscore the effectiveness of our approach, offering a comprehensive solution to the persistent challenges in FER, and paving the way for more empathetic AI, advanced human computer interfaces, and improved mental health support systems.

## Table of Contents

- [Ensemble Models For Facial Emotion Recognition](#ensemble-models-for-facial-emotion-recognition)
  - [Table of Contents](#table-of-contents)
  - [Read me first](#read-me-first)
  - [Benchmark Results](#benchmark-results)
  - [ResEmoteNet](#resemotenet)
  - [PAtt-Lite](#patt-lite)
  - [Residual Masking Network](#residual-masking-network)
  - [EmoNeXt](#emonext)
  - [Other models](#other-models)
  - [Latest Changes](#latest-changes)
  - [Other Mentions](#other-mentions)
  - [License](#license)

## Read me first

In the rest of the README you can see a table that presents the results obtained with implementations on different datasets.

Then for each of those implementations we give a brief description of what we did (our experiments) and the link to the original paper where it appeared. Each of those implementation directories have its own README where you can find instructions on how to run it.

Finally, we have some other mentions of thing I considered important to know and some additional resources.

## Benchmark Results

<img src="OtherModels\graphs\test_metrics_v2.png" alt="graph" width="600" height="350">

## ResEmoteNet

In that directory we conduct the experiments focused on the ResEmoteNet model.

**Experiments:**

We employ the base model as described in paper, we do several experiments with different hyperparameters, small changes to architecture, using ensemble approach and fine-tuning the model.

The datasets used for those experiments are: Fer2013, Fer2013_b (balanced version of fer2013, that is private), Fer+.

For more details check the Readme file from that specific directory.

**Paper Reference:**

- [link](https://arxiv.org/abs/2409.10545)

## PAtt-Lite

In that directory we conduct the experiments focused on the PAtt-Lite model.

**Experiments:**

- Changed dataset. hyperparameters and fine-tuning protocol
- Changed attention layer, translated arhitecture from Tensorflow to PyTorch
- Changed backbone model

**Paper Reference:**

- [link](https://arxiv.org/abs/2306.09626)

## Residual Masking Network

In that directory we conduct the experiments focused on the ResMaskingNet model.

**Experiments:**

- Changed dataset, changed face model, changed models within the ensemble.

**Paper Reference:**

- [link](https://ieeexplore.ieee.org/document/9411919)

## EmoNeXt

In that directory we conduct the experiments focused on the EmoNeXt model.

**Experiments:**

- Changed the backbone, loss function, hyperparameters and dataset.

**Paper Reference:**

- [link](https://www.researchgate.net/publication/374372487_EmoNeXt_an_Adapted_ConvNeXt_for_Facial_Emotion_Recognition)

## Other models

In that directory we did the experiments with the rest of well known model arhitectures. The models (taken using PyTorch models library or Timm library) were trained locally, and are the following:

- EfficientNet B0 & V2M
- Eva
- FBNetV3B
- MobileNet V3 & V4
- ResNeXt50_32x4d

Added ensemble apporach, along its evaluation and usage. Added single model usage.

Results:

<img src="OtherModels\graphs\validation_metrics.png" alt="graph" width="650" height="550">

**Paper References:**

- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Eva](https://arxiv.org/abs/2211.07636)
- [FBNetV3B](https://arxiv.org/abs/2006.02049)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [MobileNetV4](https://arxiv.org/abs/2404.10518)
- [ResNeXt50_32x4d](https://arxiv.org/abs/1611.05431)

## Latest Changes

- **2024-11-08:** Initial release with ResEmoteNet

- **2024-11-22:** Added enhanced version of ResEmoteNet (see comits)

- **2024-12-03:** Readme updates plus some fixes to paper

- **2024-12-04:** Added PAtt-Lite approach and fixed it to run

- **2024-12-08:** Added experiments for PAtt-Lite and final version

- **2024-12-11:** Added datasets experiments and embeddings experiments

- **2024-12-12:** Added final ResEmoteNet with good results

- **2024-12-13:** Added EmoNeXt base version

- **2024-12-15:** Added final EmoNeXt version

- **2024-12-16:** Added Residual Masking Network base version

- **2024-12-18:** Added final Residual Masking Network version

- **2025-01-09:** Added final Residual Masking Network version

- **2025-01-10:** Added other models part 1

- **2025-01-13:** Added other models part 2 finall + ensemble + use

## Other Mentions

 **Future Work:**

 Plans for future work include exploring other ensemble methods and incorporating additional datasets.

**Plan for Work**

Extend the dataset and test results on it (Weeks 1-4).

Find optimization for training and methods to quantize the models if needed and evaluate new models (Weeks 5-8).

Prepare the final paper and the presentation (Weeks 9-12).

 **Related Resources:**

- [Kaggle link for extra experiments](https://www.kaggle.com/code/alexandruberindeie/fer-workspace)

## License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
