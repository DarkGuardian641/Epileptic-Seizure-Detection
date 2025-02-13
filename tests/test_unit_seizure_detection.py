# test_seizure_detection.py

import pytest
from tests.test_seizure_detection import detect_seizure

# Test case 1: Test seizure detection with signals above threshold (seizure detected)
def test_seizure_detection_with_valid_data():
    eeg_signal = [500, 1000, 1500, 800, 1200]  # Contains a signal above the threshold
    result = detect_seizure(eeg_signal)
    assert result == True, "The system should detect a seizure."

# Test case 2: Test with signals below threshold (no seizure)
def test_no_seizure_with_valid_data():
    eeg_signal = [500, 400, 300, 200, 100]  # All signals below threshold
    result = detect_seizure(eeg_signal)
    assert result == False, "The system should not detect a seizure."

# Test case 3: Test with an empty signal list (raises error)
def test_empty_signal():
    with pytest.raises(ValueError, match="Input must be a non-empty list of EEG signals"):
        detect_seizure([])

# Test case 4: Test with a non-list input (raises error)
def test_non_list_input():
    with pytest.raises(ValueError, match="Input must be a non-empty list of EEG signals"):
        detect_seizure("not a list")

# Test case 5: Test with invalid signal data (e.g., strings within EEG signal)
def test_invalid_signal_type():
    eeg_signal = [100, 200, "invalid", 400]  # Invalid signal data (string)
    with pytest.raises(ValueError, match="Input must be a non-empty list of EEG signals"):
        detect_seizure(eeg_signal)

# Test case 6: Test with multiple peaks exceeding threshold (seizure detected)
def test_multiple_peaks():
    eeg_signal = [10, 2000, 500, 1500, 800]  # Contains multiple peaks above the threshold
    result = detect_seizure(eeg_signal)
    assert result == True, "The system should detect a seizure."

# Test case 7: Test with all signals below the threshold (no seizure)
def test_signal_without_peaks():
    eeg_signal = [100, 200, 300, 400]  # No signals exceed the threshold
    result = detect_seizure(eeg_signal)
    assert result == False, "The system should not detect a seizure."