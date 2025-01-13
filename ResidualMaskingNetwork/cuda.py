import json
import numpy as np
from utils.datasets.fer2013dataset import fer2013

def save_test_targets():
    # Load configuration
    with open("D:\\Alex\\Desktop\\ResidualMaskingNetwork-master\\configs\\fer2013_config.json") as f:
        configs = json.load(f)

    # Load test dataset
    test_set = fer2013("test", configs, tta=False)  # Set tta to False to get individual samples

    # Extract labels
    test_targets = []
    for idx in range(len(test_set)):
        _, target = test_set[idx]
        test_targets.append(target)

    # Save labels to .npy file
    np.save("saved/test_targets.npy", test_targets)
    print("Test targets saved successfully.")

if __name__ == "__main__":
    save_test_targets()
