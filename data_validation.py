import numpy as np

def detect_nan_inf(data):
    """
    Detect NaN or Inf values in EEG data.

    INPUT:
        data: np.ndarray, shape (channels, samples)
            Raw EEG data from board

    OUTPUT:
        is_valid: bool
            True if no NaN/Inf found
        nan_channels: list
            Indices of channels containing NaN/Inf

    EXAMPLE:
        >>> data = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
        >>> is_valid, bad_ch = detect_nan_inf(data)
        >>> print(is_valid)  # False
        >>> print(bad_ch)    # [0]
    """
    # YOUR IMPLEMENTATION HERE
    # Hint: Use np.isnan() and np.isinf()

    # create list of channels containing nan/inf
    nan_channels = []
    is_valid = True

    invalid_check = np.isnan(data) | np.isinf(data)
    for channel_idx, channel in enumerate(invalid_check):
        if channel.any():
            is_valid = False
            nan_channels.append(channel_idx)
            # skip rest of data if nan/inf found
            continue
    
    return is_valid, nan_channels
    # pass

def detect_flatline(data, threshold=0.1):
    """
    Detect flatline channels (constant/near-constant signal).

    INPUT:
        data: np.ndarray, shape (channels, samples)
            EEG data segment
        threshold: float
            Standard deviation threshold in microvolts (default: 0.1 uV)

    OUTPUT:
        flatline_channels: list
            Indices of flatline channels
        channel_stds: np.ndarray
            Standard deviation of each channel

    REASONING:
        Healthy EEG typically has std > 1 uV.
        Flatline may indicate an electrode is disconnected or broken.
    """
    # YOUR IMPLEMENTATION HERE
    # Hint: Calculate std per channel, compare to threshold
    flatline_channels = []
    channel_stds = []

    for channel_idx, channel in enumerate(data, dtype = float):
        # compute std for non-nan elements in channel
        s = np.nanstd(channel)
        channel_stds.append(s)
        if s < threshold:
            flatline_channels.append(channel_idx)

    return flatline_channels, channel_stds
    # pass

def detect_extreme_noise(data, z_threshold=5.0):
    """
    Detect extreme noise/spike artifacts.

    INPUT:
        data: np.ndarray, shape (channels, samples)
        z_threshold: float
            Z-score threshold for spike detection (default: 5.0)

    OUTPUT:
        has_spikes: bool
            True if spikes detected in any channel
        spike_channels: list
            Channels containing spikes
        spike_percentage: float
            Percentage of samples that are spikes

    DETECTION LOGIC:
        For each channel:
        1. Calculate mean and std
        2. Compute z-scores: z = (x - mean) / std
        3. Flag samples where |z| > threshold
        4. If > 1% of samples are spikes, mark channel bad
    """
    # YOUR IMPLEMENTATION HERE
    has_spikes = False
    spike_channels = []
    total_spike_count = 0
    total_valid_count = 0

    data = np.asarray(data, dtype=float)

    for channel_idx, channel in enumerate(data):
        # calculate mean and std
        mean = np.nanmean(channel)
        std = np.nanstd(channel)
        # filter invalid data
        valid_data = np.isfinite(data)

        channel_spike_count = 0

        # channel has flat or invalid std
        if not np.isfinite(std) or std == 0:
            continue
            # channel_spike_count = 0

        # loop through all data points
        for x in channel:
            # compute z score
            z_score = (x - mean) / std
            if abs(z_score) > z_threshold:
                channel_spike_count += 1
        # data has spikes
        if channel_spike_count > 0:
            has_spikes = True
        # mark channel as bad
        percent = channel_spike_count / len(channel)
        if percent > 0.01:
            spike_channels.append(channel_idx)
        spike_count += channel_spike_count
        data_count += len(channel)

    return has_spikes, spike_channels, spike_count / data_count * 100.0

    # pass

def detect_channel_duplication(data, correlation_threshold=0.99):
    """
    Detect if channels are duplicates of each other.

    INPUT:
        data: np.ndarray, shape (channels, samples)
        correlation_threshold: float
            Pearson correlation threshold (default: 0.99)

    OUTPUT:
        duplicate_pairs: list[tuple]
            [(ch_i, ch_j), ...] duplicate channel pairs
        correlation_matrix: np.ndarray, shape (channels, channels)
            Full correlation matrix

    USAGE:
        If duplicates are found, the data is invalid — likely a hardware issue.
    """
    # YOUR IMPLEMENTATION HERE
    # Hint: Use np.corrcoef(data) to get the correlation matrix

    data = np.asarray(data, dtype=float)

    duplicate_pairs = []

    # creates square matrix with corr coeffs between i,j
    correlation_matrix = np.corrcoef(data)
    n = len(correlation_matrix[0])

    # only iterate through upper triangular
    for row in range(n):
        for col in range(row + 1, n):
            if abs(correlation_matrix[row][col]) >= correlation_threshold:
                duplicate_pairs.append((row, col))
    
    return duplicate_pairs, correlation_matrix
        
    # pass

class ValidationPackage:
    """
    Container for validated EEG data with quality metrics.

    ATTRIBUTES:
        data: np.ndarray
            The validated EEG data
        is_valid: bool
            Overall validity flag
        timestamp: float
            Timestamp when data was collected
        quality_metrics: dict
            Dictionary of quality measurements:
            {
                'has_nan': bool,
                'has_inf': bool,
                'flatline_channels': list,
                'noisy_channels': list,
                'duplicate_pairs': list,
                'sampling_rate': float
            }

    USAGE:
        package = ValidationPackage(data, timestamp)
        if package.is_valid:
            # Safe to process
            process(package.data)
        else:
            # Log the issue
            log_error(package.quality_metrics)
    """
    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp
        self.is_valid = True
        self.quality_metrics = {}
