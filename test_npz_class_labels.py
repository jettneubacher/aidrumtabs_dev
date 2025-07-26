# test class labels in npz files

import numpy as np
import os

# Update label_map with all your class names
label_map = {
    'snare': 0,
    'bass': 1,
    'hi-hat': 2,
    'tom': 3,
    'cymbal': 4,
    'unknown': -1  # For any unknown annotations
}

def process_annotations(annotations):
    """
    Process annotations to map them to the correct label using the label_map.
    Returns a list of labels, replacing unknowns with 'unknown' label.
    """
    processed_annotations = []
    unknown_found = False  # Track if any unknown annotations are found
    for annotation in annotations:
        # Extract the label (second element) from the annotation
        label = annotation[1]  # annotation[0] is timestamp, annotation[1] is label
        
        # Handle the label using the label_map
        label = label_map.get(label, label_map['unknown'])
        
        if label == label_map['unknown']:
            unknown_found = True
        processed_annotations.append(label)
    
    return processed_annotations, unknown_found

def test_npz_file(npz_file_path):
    """
    Loads the .npz file, extracts features and annotations,
    then processes annotations to check if any are unknown.
    """
    data = np.load(npz_file_path)

    # Extract features and annotations
    mfccs = data['mfccs']
    onset_env = data['onset_env']
    spectrogram = data['spectrogram']
    annotations = data['annotations']

    # Print the type and sample of the annotations
    #print(f"Annotations type in {npz_file_path}: {type(annotations)}")
    #print(f"Sample annotations data: {annotations[:5]}")  # Print the first 5 annotations for inspection

    # Process annotations and check for unknowns
    processed_annotations, unknown_found = process_annotations(annotations)
    
    if unknown_found:
        print(f"Warning: Unknown annotations found in {npz_file_path}")

def test_all_npz_files(directory_path):
    """
    Loops through all .npz files in the directory and tests each one.
    """
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".npz"):
            npz_file_path = os.path.join(directory_path, file_name)
            test_npz_file(npz_file_path)

if __name__ == "__main__":
    # Path to directory containing the .npz files
    npz_directory = "data/enst_drums/processed_features/processed_features3"
    
    # Test all .npz files in the directory
    test_all_npz_files(npz_directory)
