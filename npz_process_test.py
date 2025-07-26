# process ONE npz file into a np array

import numpy as np
import os


def process_single_npz_file(npz_path, timeframe_length_s=0.116, feature_frames_per_timeframe=10):
    data = np.load(npz_path, allow_pickle=True)
    
    mfccs = data['mfccs']  # Shape: (13, total_frames)
    spectrogram = data['spectrogram']  # Shape: (1025, total_frames)
    onset_env = data['onset_env']  # Shape: (total_frames,)
    annotations = data['annotations']  # Shape: (N, 2) - timestamps & drum labels
    
    total_feature_frames = mfccs.shape[1]
    total_timeframes = total_feature_frames // feature_frames_per_timeframe
    
    print(f"Total feature frames: {total_feature_frames}")
    print(f"Total timeframes: {total_timeframes}")
    
    # Label mapping
    class_map = {"snare": 1, "hi-hat": 3, "cymbal": 4, "bass": 0, "tom": 2}
    
    # Initialize X and Y
    X = []
    Y = []
    
    for timeframe_index in range(total_timeframes):
        start_idx = timeframe_index * feature_frames_per_timeframe
        end_idx = start_idx + feature_frames_per_timeframe
        
        mfcc_frame = mfccs[:, start_idx:end_idx]
        spectrogram_frame = spectrogram[:, start_idx:end_idx]
        onset_env_frame = onset_env[start_idx:end_idx].reshape(1, -1)
        
        feature_vector = np.vstack([mfcc_frame, spectrogram_frame, onset_env_frame])
        X.append(feature_vector)
        
        # Initialize label vector as all zeros
        label_vector = np.zeros(len(class_map))
        
        # Convert timeframe index to actual time range
        timeframe_start = timeframe_index * timeframe_length_s
        timeframe_end = timeframe_start + timeframe_length_s
        
        # Process annotations
        for annotation in annotations:
            timestamp = float(annotation[0])  # Convert timestamp to float
            drum_class = annotation[1]
            
            if timeframe_start <= timestamp < timeframe_end:
                if drum_class in class_map:
                    label_vector[class_map[drum_class]] = 1
                
        Y.append(label_vector)
        
    # Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Final X shape: {X.shape}")
    print(f"Final Y shape: {Y.shape}")
    
    np.savez("process_test2.npz", X=X, Y=Y)

    return X, Y

# Example usage
npz_file = "data/enst_drums/processed_features/processed_features1/123_MIDI-minus-one_blues-102_sticks_features.npz"  # Change to actual NPZ file path
X, Y = process_single_npz_file(npz_file)
