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

def load_raw_covid_data(limit=2000):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    covid_path = os.path.join(base_dir, "data", "covid")
    non_covid_path = os.path.join(base_dir, "data", "noncovid")

    # Define limits based on the ratio
    covid_limit = int(0.5 * limit)  
    non_covid_limit = limit - covid_limit  #
    def process_images(path, limit):
        images = [f for f in os.listdir(path) if f.endswith('.png')][:limit]
        data = np.empty((len(images), 128, 128, 1), dtype=np.float32)

        for i, f in enumerate(images):
            full_path = os.path.join(path, f)
            try:
                print(f"Reading: {full_path}")
                img_data = img.imread(full_path)
                data[i] = resize(img_data, (128, 128, 1), anti_aliasing=True)
            except Exception as e:
                print(f"❌ Error reading file: {full_path}")
                print(f"   ↳ {type(e).__name__}: {e}")
                continue  # or continue if you want to skip bad files

        return data


    # Load images based on the new limits
    covid_images = process_images(covid_path, covid_limit)
    non_covid_images = process_images(non_covid_path, non_covid_limit)

    X = np.concatenate([covid_images, non_covid_images])
    y = np.array([1] * len(covid_images) + [0] * len(non_covid_images), dtype=np.float32)
    
    return X, y



