import io
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import streamlit as st

# --- Core Audio Processing Functions ---

def apply_reverb(y, sr, wet_level=0.3, delay_ms=40, decay=0.5):
    """
    Applies a simple feedback delay network to simulate room reverb.
    wet_level: 0.0 (Dry) to 1.0 (100% Reverb)
    delay_ms: delay time in milliseconds
    decay: feedback decay factor (0.1 to 0.8)
    """
    if wet_level <= 0:
        return y
    
    delay_samples = int(sr * (delay_ms / 1000.0))
    output = np.copy(y)
    
    # Process each channel if stereo, or 1D if mono
    if output.ndim == 1:
        for i in range(delay_samples, len(output)):
            output[i] += output[i - delay_samples] * decay
        return (1 - wet_level) * y + wet_level * output
    else:
        for ch in range(output.shape[0]):
            for i in range(delay_samples, output.shape[1]):
                output[ch, i] += output[ch, i - delay_samples] * decay
        return (1 - wet_level) * y + wet_level * output

def apply_bass_boost(y, sr, gain_db=6.0, cutoff=200):
    """Custom low-shelf filter for bass boosting."""
    if gain_db == 0:
        return y
    # Low pass filter design via scipy
    b, a = scipy.signal.butter(2, cutoff / (sr / 2), btype='low')
    gain_linear = 10 ** (gain_db / 20)
    filtered = scipy.signal.lfilter(b, a, y)
    return y + (filtered * (gain_linear - 1))


# --- Streamlit Interface ---

st.title("CA.Editor — Web-Based Audio Studio")
st.write("Lightweight, in-memory real-time audio manipulation.")

uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Read audio into memory only once
    if "audio_data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        y, sr = librosa.load(uploaded_file, sr=None, mono=False)
        st.session_state["audio_data"] = y
        st.session_state["sr"] = sr
        st.session_state["file_name"] = uploaded_file.name

    y = st.session_state["audio_data"]
    sr = st.session_state["sr"]

    st.subheader("Audio Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speed Modifier", 0.5, 2.0, 1.0, 0.05)
        pitch = st.slider("Pitch Shift (Semitones)", -8, 8, 0, 1)
    with col2:
        bass = st.slider("Bass Boost (dB)", 0, 12, 0, 1)
        reverb = st.slider("Reverb (Wet/Dry Mix)", 0.0, 0.8, 0.0, 0.05)

    # --- Processing Pipeline ---
    processed_y = y.copy()

    # 1. Pitch Shift
    if pitch != 0:
        processed_y = librosa.effects.pitch_shift(y=processed_y, sr=sr, n_steps=pitch)

    # 2. Speed Modifier
    if speed != 1.0:
        if processed_y.ndim > 1:
            # Handle multi-channel speed modification
            channels = [librosa.effects.time_stretch(y=processed_y[c], rate=speed) for c in range(processed_y.shape[0])]
            processed_y = np.array(channels)
        else:
            processed_y = librosa.effects.time_stretch(y=processed_y, rate=speed)

    # 3. Bass Boost
    if bass > 0:
        processed_y = apply_bass_boost(processed_y, sr, gain_db=bass)

    # 4. Reverb Section
    if reverb > 0:
        processed_y = apply_reverb(processed_y, sr, wet_level=reverb)

    # Export to BytesIO buffer
    buffer = io.BytesIO()
    # Format array back to standard channel dimensions if needed for SoundFile
    out_data = processed_y.T if processed_y.ndim > 1 else processed_y
    sf.write(buffer, out_data, sr, format='WAV')
    buffer.seek(0)

    st.subheader("Preview & Download")
    st.audio(buffer, format="audio/wav")
    st.download_button("Download Processed Audio", data=buffer, file_name="CA_Editor_Output.wav", mime="audio/wav")
                                                     
