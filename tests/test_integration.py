import pytest
from tests.test_seizure_detection import detect_seizure

def test_integration_seizure_detected():
    """
    Test integration of detect_seizure function with a full system scenario
    where high EEG values trigger seizure detection.
    """
    eeg_signal = [100, 200, 1200, 900, 1100]  # Values above threshold
    result = detect_seizure(eeg_signal)
    assert result == True, "Seizure should be detected in this scenario."

def test_integration_no_seizure():
    """
    Test integration where all values are below the seizure threshold.
    """
    eeg_signal = [100, 200, 300, 400, 500]  # All values below threshold
    result = detect_seizure(eeg_signal)
    assert result == False, "Seizure should not be detected in this scenario."

def test_integration_invalid_input():
    """
    Test integration handling of invalid inputs.
    """
    with pytest.raises(ValueError, match="Input must be a non-empty list of EEG signals"):
        detect_seizure(["invalid", 300, 500])
