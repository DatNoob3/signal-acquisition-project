# Add after existing imports
from data_validation import (
    detect_nan_inf,
    detect_flatline,
    detect_extreme_noise,
    detect_channel_duplication,
    ValidationPackage
)

# Current line 151:
eeg_data = data[self.eeg_channels, :]

# INSERT AFTER LINE 151:
# Validate new data chunk
is_valid, nan_channels = detect_nan_inf(eeg_data)
if not is_valid:
    print(f"Warning: NaN/Inf detected in channels {nan_channels}")
    # Skip this chunk or use last known good data
    return self.processed_data_buffer[:, -int(duration * self.sampling_rate):]

# Current line 177 returns model_input
# INSERT BEFORE RETURN:

# Comprehensive validation on buffer
flatline_ch = detect_flatline(recent_data)
if len(flatline_ch) > 0:
    print(f"Warning: Flatline detected in channels {flatline_ch}")

has_spikes, spike_ch, spike_pct = detect_extreme_noise(recent_data)
if has_spikes and spike_pct > 5.0:
    print(f"Warning: Excessive noise ({spike_pct:.1f}%) in channels {spike_ch}")

# Then continue with existing return statement