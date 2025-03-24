import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from matplotlib import image as img
import numpy as np
import tensorflow as tf

# Data Loading Function
import os
import numpy as np
from matplotlib import image as img
from skimage.transform import resize

def load_raw_covid_data_for_federated(
    num_clients=2,
    val_split=0.1,
    limit_per_client=400,
    covid_ratio=0.8,
    allow_noncovid_overlap=True,
    seed=42
):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    covid_path = os.path.join(base_dir, "data", "covid")
    non_covid_path = os.path.join(base_dir, "data", "noncovid")

    covid_images = [os.path.join(covid_path, f) for f in os.listdir(covid_path) if f.endswith('.png')]
    non_covid_images = [os.path.join(non_covid_path, f) for f in os.listdir(non_covid_path) if f.endswith('.png')]

    np.random.seed(seed)
    np.random.shuffle(covid_images)
    np.random.shuffle(non_covid_images)

    # ✅ Fixed test set size: 200 COVID, 100 non-COVID
    covid_test = covid_images[:200]
    noncovid_test = non_covid_images[:100]

    covid_pool = covid_images[200:]
    noncovid_pool = non_covid_images[100:]

    client_data = []

    # Determine client-specific limits
    if limit_per_client is not None:
        covid_per_client = int(limit_per_client * covid_ratio)
        noncovid_per_client = limit_per_client - covid_per_client
    else:
        covid_per_client = len(covid_pool) // num_clients
        noncovid_per_client = len(noncovid_pool) if allow_noncovid_overlap else len(noncovid_pool) // num_clients

    # Split COVID pool evenly
    covid_splits = np.array_split(covid_pool[:covid_per_client * num_clients], num_clients)

    if allow_noncovid_overlap:
        noncovid_subset = noncovid_pool[:noncovid_per_client]
        for i in range(num_clients):
            client_files = list(covid_splits[i]) + list(noncovid_subset)
            labels = [1] * len(covid_splits[i]) + [0] * len(noncovid_subset)
            combined = list(zip(client_files, labels))
            np.random.shuffle(combined)
            client_data.append(combined)
    else:
        noncovid_splits = np.array_split(noncovid_pool[:noncovid_per_client * num_clients], num_clients)
        for i in range(num_clients):
            client_files = list(covid_splits[i]) + list(noncovid_splits[i])
            labels = [1] * len(covid_splits[i]) + [0] * len(noncovid_splits[i])
            combined = list(zip(client_files, labels))
            np.random.shuffle(combined)
            client_data.append(combined)

    # Server test set (fixed size)
    test_files = covid_test + noncovid_test
    test_labels = [1] * len(covid_test) + [0] * len(noncovid_test)

    return client_data, (test_files, test_labels)


         

def load_images_from_paths(image_label_pairs):
    data = []
    labels = []
    for path, label in image_label_pairs:
        try:
            img_data = img.imread(path)
            img_data = resize(img_data, (128, 128, 1), anti_aliasing=True)
            data.append(img_data)
            labels.append(label)
        except Exception as e:
            print(f"❌ Error reading file: {path} ({e})")
            continue
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

