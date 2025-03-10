import os

def count_files(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Example usage
folder_path = "/home/jmulvany/thesis_implementation/Thesis/data/noncovid"  # Change this to your folder path
print(f"Number of files in '{folder_path}': {count_files(folder_path)}")
