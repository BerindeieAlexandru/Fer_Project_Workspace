import itertools
import json
import random

import imgaug
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# for consistent latex font
from matplotlib import rc
from sklearn.metrics import confusion_matrix

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from tqdm import tqdm

from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()


checkpoint_name = "resnet34_test_2024May28_12.14"


def main():
    with open("D:\\Alex\\Desktop\\ResidualMaskingNetwork-master\\configs\\fer2013_config.json") as f:
        configs = json.load(f)

    # state = torch.load("D:\\Alex\\Desktop\\ResidualMaskingNetwork-master\\checkpoint\\".format(checkpoint_name))
    state = torch.load("D:\\Alex\\Desktop\\ResidualMaskingNetwork-master\\checkpoint\\resnet34_test_2024May28_12.14")

    from models import cbam_resnet50
    from models import alexnet
    from models import resnet34

    model = cbam_resnet50

    model = model(in_channels=3, num_classes=7).cuda()
    model.load_state_dict(state["net"])
    model.eval()

    correct = 0
    total = 0
    all_target = []
    all_output = []

    test_set = fer2013("test", configs, tta=True, tta_size=8)
    # test_set = fer2013('test', configs, tta=False, tta_size=0)

    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]

            images = make_batch(images)
            images = images.cuda(non_blocking=True)

            outputs = model(images).cpu()
            outputs = F.softmax(outputs, 1)

            # outputs.shape [tta_size, 7]
            outputs = torch.sum(outputs, 0)
            outputs = torch.argmax(outputs, 0)
            outputs = outputs.item()
            targets = targets.item()
            total += 1
            correct += outputs == targets

            all_target.append(targets)
            all_output.append(outputs)

    # acc = 100. * correct / total
    # print("Accuracy {:.03f}".format(acc))

    all_target = np.array(all_target)
    all_output = np.array(all_output)

    matrix = confusion_matrix(all_target, all_output)
    np.set_printoptions(precision=2)

    # plt.figure(figsize=(5, 5))
    plot_confusion_matrix(
        matrix,
        classes=class_names,
        normalize=True,
        # title='{} \n Accuracc: {:.03f}'.format(checkpoint_name, acc)
        title="CBAM Resnet 50",
    )

    # plt.show()
    # plt.savefig('cm_{}.png'.format(checkpoint_name))
    plt.savefig("./saved/cm/cm_{}.pdf".format(checkpoint_name))
    plt.close()


if __name__ == "__main__":
    main()
