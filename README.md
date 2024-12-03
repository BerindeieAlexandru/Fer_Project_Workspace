# Ensemble Models For Facial Emotion Recognition

**Project description:**

Facial Expression Recognition (FER) is a significant machine learning challenge, focused on the accurate identification of human facial expressions. Despite recent advancements, enhancing performance and obtaining reliable results in real-world conditions continues to pose substantial difficulties.

In this REPOSITORY, we will present several FER approaches, focusing on FER2013 and FER+ datasets, a widely used for this task.

In the provided paper we present the problem that we are trying to solve then we present the theoretical background talking about related work, the notions used and about the dataset. We then present the methodology of research that we used to select the papers and the methods used. We present five popular and effective approaches focusing on their contribution and the results, some of those being ResEmoteNet, PAtt-Lite, Residual Masking Network.

We implement them and try to improve the presented results using different augmentations on the dataset, changing hyper-parameters, employing an ensemble approach (composed of the best performing models) and even fine-tuning the models. Finally, we evaluate our method by comparing the results with leading models in the field, demonstrating a significant increase in accuracy.

The results underscore the effectiveness of our approach, offering a comprehensive solution to the persistent challenges in FER, and paving the way for more empathetic AI, advanced human computer interfaces, and improved mental health support systems.

## Table of Contents

- [Latest Changes](#latest-changes)
- [Read me first](#read-me-first)
- [Benchmark Results](#benchmark-results)
- [ResEmoteNet](#resemotenet)
- [PAtt-Lite](#patt-lite)
- [Residual Masking Network](#residual-masking-network)
- [Other Mentions](#other-mentions)
- [License](#license)

## Latest Changes

- **2024-11-08:** Initial release with ResEmoteNet

- **2024-11-22:** Added final version of ResEmoteNet (see comits)

- **2024-12-03:** Readme updates plus some fixes to paper

## Read me first

In the rest of the README you can see a table that presents the results obtained with implementations on different datasets.

Then for each of those implementations we give a brief description of what we did (our experiments) and the link to the original paper where it appeared. Each of those implementation directories have its own README where you can find instructions on how to run it.

Finally, we have some other mentions of thing I considered important to know and some additional resources.

## Benchmark Results

| Model                    | Dataset | Ensemble | Accuracy (%) | Finetune |
| ------------------------ | :----------: | :----------: | :-------: | :----------: |
| ResEmoteNet              |     Fer2013     |     -     |    61.53   |     -     |
| ResEmoteNet              |     Fer2013_b     |     -     |    74.70   |     -     |
| PAtt-Lite                |     XX.X     |     XX.X     |    XX.X   |     XX.X     |
| Residual Masking Network |     XX.X     |     XX.X     |    XX.X   |     XX.X     |

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

To be filled.

**Paper Reference:**

- [link](https://arxiv.org/abs/2306.09626)

## Residual Masking Network

In that directory we conduct the experiments focused on the ResMaskingNet model.

**Experiments:**

To be filled.

**Paper Reference:**

- [link](https://ieeexplore.ieee.org/document/9411919)

## Other Mentions

 **Future Work:**

 Plans for future work include exploring other ensemble methods and incorporating additional datasets.

 **Related Resources:**

- [Kaggle link for extra experiments](https://www.kaggle.com/code/alexandruberindeie/fer-workspace)

### License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
