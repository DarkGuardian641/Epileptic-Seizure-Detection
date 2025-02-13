def detect_seizure(eeg_signal):
    """
    Detects whether a seizure is occurring based on EEG signal data.
    A seizure is detected if any value in the EEG signal exceeds a defined threshold.
    
    Parameters:
        eeg_signal (list): A list of numerical EEG signal values.
    
    Returns:
        bool: True if a seizure is detected, False otherwise.
    
    Raises:
        ValueError: If the input is not a non-empty list of numerical values.
    """
    # Define a threshold for seizure detection
    SEIZURE_THRESHOLD = 1000
    
    # Validate input
    if not isinstance(eeg_signal, list) or len(eeg_signal) == 0:
        raise ValueError("Input must be a non-empty list of EEG signals")
    
    if not all(isinstance(value, (int, float)) for value in eeg_signal):
        raise ValueError("Input must be a non-empty list of EEG signals")
    
    # Check if any value in the signal exceeds the threshold
    return any(value > SEIZURE_THRESHOLD for value in eeg_signal)
