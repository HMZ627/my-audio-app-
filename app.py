import io
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import streamlit as st

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CA.Editor — Web Audio Studio",
    page_icon="🎧",
    layout="wide"
)

# --- Define Your Transparent Headset Asset ---
# IMPORTANT: This MUST be a PNG with a transparent background.
# I have found a temporary, clean example that works well with dark backgrounds.
HEADSET_URL = "https://freepngimg.com/download/headphones/2-headphones-png-image-with-transparency-background.png"

# --- Inject Custom CSS for Glassmorphism & Continuous Rotation ---
# This CSS removes the yellow box and makes the image float seamlessly.
custom_css = f"""
<style>
/* Main App Background - Dark Studio Theme */
.stApp {{
    background-color: #01060a; /* Pure deep black/dark navy */
    color: #f0f6fc;
}}

/* -- GLASSMORPHISM UI STYLING -- */
/* Makes the content panels slightly translucent with a blur effect */
div.block-container {{
    background: rgba(13, 17, 23, 0.7); /* Translucent dark charcoal */
    backdrop-filter: blur(10px); /* The "Glass" effect */
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 2rem !important;
    margin-top: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    z-index: 2; /* Keeps UI above the headset */
    position: relative;
}}

/* -- BACKGROUND ASSET STYLING -- */
/* Handles the rotating transparent headset */
.headset-container {{
    position: fixed;
    top: 50%;
    right: 5%; /* Positioned like the mockup */
    transform: translateY(-50%);
    width: 500px;
    height: 500px;
    z-index: 1; /* Sits behind the Glassmorphism UI */
    pointer-events: none; /* User can click "through" it to UI elements */
    opacity: 0.8; /* Subtle presence */
}}

.rotating-headset {{
    width: 100%;
    height: 100%;
    background-image: url('{HEADSET_URL}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    /* Continuous Z-Axis Rotation */
    animation: rotateSeamless 30s linear infinite;
    /* Subtle Cyan Glow matching Mockup data trails */
    filter: drop-shadow(0 0 25px rgba(0, 229, 255, 0.5));
}}

/* Define the continuous rotation keyframes */
@keyframes rotateSeamless {{
    from {{
        transform: rotate(0deg);
    }}
    to {{
        transform: rotate(360deg);
    }}
}}

/* -- STYLING STREAMLIT COMPONENTS -- */
/* Custom styling for sliders and buttons to match the aesthetic */
.stSlider > div > div > div > div {{
    background-color: #00e5ff; /* Cyan accent */
}}

.stSlider > div {{
    color: #ffffff;
}

.stButton > button {{
    background: linear-gradient(135deg, #00e5ff 0%, #0077ff 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.7);
    transform: translateY(-2px);
}}
</style>

<!-- Background Headset HTML Container -->
<div class="headset-container">
    <div class="rotating-headset"></div>
</div>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# --- Core Audio Processing Functions (Unchanged) ---

def apply_reverb(y, sr, wet_level=0.3, delay_ms=40, decay=0.5):
    """Applies a feedback delay network to simulate audio reverb."""
    if wet_level <= 0: return y
    delay_samples = int(sr * (delay_ms / 1000.0))
    output = np.copy(y)
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
    if gain_db == 0: return y
    b, a = scipy.signal.butter(2, cutoff / (sr / 2), btype='low')
    gain_linear = 10 ** (gain_db / 20)
    filtered = scipy.signal.lfilter(b, a, y)
    return y + (filtered * (gain_linear - 1))

# --- App Layout & Logic ---

# --- App Header ---
st.title("CA.Editor — Web-Based Audio Studio")
st.write("Lightweight, in-memory real-time audio manipulation engine.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload an Audio File (WAV, MP3, OGG, FLAC)", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Read audio array into session memory once
    if "audio_data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Uploading and decoding audio array..."):
            y, sr = librosa.load(uploaded_file, sr=None, mono=False)
            st.session_state["audio_data"] = y
            st.session_state["sr"] = sr
            st.session_state["file_name"] = uploaded_file.name

    y = st.session_state["audio_data"]
    sr = st.session_state["sr"]

    st.subheader("Studio Controls")
    
    # Interactive Controls Layout
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speed Modifier", 0.5, 2.0, 1.0, 0.05)
        pitch = st.slider("Pitch Shift (Semitones)", -8, 8, 0, 1)
    with col2:
        bass = st.slider("Bass Boost (dB)", 0, 12, 0, 1)
        reverb = st.slider("Reverb (Wet/Dry Mix)", 0.0, 0.8, 0.0, 0.05)

    # Audio Pipeline Processing
    with st.spinner("Applying digital signal processing..."):
        processed_y = y.copy()

        # 1. Pitch Shift
        if pitch != 0:
            processed_y = librosa.effects.pitch_shift(y=processed_y, sr=sr, n_steps=pitch)

        # 2. Speed Modifier
        if speed != 1.0:
            if processed_y.ndim > 1:
                channels = [librosa.effects.time_stretch(y=processed_y[c], rate=speed) for c in range(processed_y.shape[0])]
                processed_y = np.array(channels)
            else:
                processed_y = librosa.effects.time_stretch(y=processed_y, rate=speed)

        # 3. Bass Boost
        if bass > 0:
            processed_y = apply_bass_boost(processed_y, sr, gain_db=bass)

        # 4. Reverb Filter
        if reverb > 0:
            processed_y = apply_reverb(processed_y, sr, wet_level=reverb)

        # Export to in-memory BytesIO WAV buffer
        buffer = io.BytesIO()
        out_data = processed_y.T if processed_y.ndim > 1 else processed_y
        sf.write(buffer, out_data, sr, format='WAV')
        buffer.seek(0)

    st.subheader("Preview & Export")
    st.audio(buffer, format="audio/wav")
    st.download_button(
        label="Download Processed Audio",
        data=buffer,
        file_name="CA_Editor_Output.wav",
        mime="audio/wav"
            )
            
