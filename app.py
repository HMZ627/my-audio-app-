import streamlit as st
import librosa
import soundfile as sf
import io
import numpy as np
from scipy.signal import lfilter

# 1. Custom Website Branding & Welcome Text
st.set_page_config(page_title="CA.Editor", page_icon="🎵")
st.title("🎵 CA.Editor")
st.markdown("### Welcome to CA.Editor — Your Free Online Audio Processing Studio!")
st.write("Upload any audio file below. Move the sliders to hear changes applied automatically in real time! All processing happens securely in temporary memory.")

# 2. Upload Box
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Use session state to cache the original file array so it doesn't reload on every slider move
    if "orig_audio" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Loading audio file engine..."):
            y, sr = librosa.load(uploaded_file, sr=None)
            st.session_state.orig_audio = y
            st.session_state.sr = sr
            st.session_state.file_name = uploaded_file.name
    
    y = st.session_state.orig_audio
    sr = st.session_state.sr

    st.markdown("---")
    st.markdown("#### 🎛️ Live Audio Controls")
    
    # 3. Reactive Sliders (Changing these automatically re-runs the code below)
    speed_rate = st.slider("🏃 Speed Modifier", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="0.5x is slow motion, 2.0x is fast forward.")
    pitch_steps = st.slider("🗣️ Pitch Changer (Semitones)", min_value=-8, max_value=8, value=0, step=1, help="Negative values make the voice deeper; positive values make it higher.")
    bass_boost_db = st.slider("🔊 Bass Booster (dB)", min_value=0, max_value=15, value=0, step=1, help="Increase to pump up the low-end bass frequencies.")
    
    # 4. Live Background Processing Block
    with st.spinner("Updating live audio stream..."):
        processed_y = y.copy()
        
        # A. Live Pitch Change
        if pitch_steps != 0:
            processed_y = librosa.effects.pitch_shift(processed_y, sr=sr, n_steps=pitch_steps)
            
        # B. Live Speed Change
        if speed_rate != 1.0:
            processed_y = librosa.effects.time_stretch(processed_y, rate=speed_rate)
            
        # C. Live Bass Boost
        if bass_boost_db > 0:
            f0 = 200.0  
            w0 = f0 / (sr / 2)
            gain = 10 ** (bass_boost_db / 20)
            
            b = [1, w0 * (gain - 1)]
            a = [1, w0 * (1/gain - 1)]
            
            processed_y = lfilter(b, a, processed_y)
            
            # Normalize to prevent distortion
            max_val = np.max(np.abs(processed_y))
            if max_val > 0:
                processed_y = processed_y / max_val

        # 5. Stream the output matrix back into RAM bytes buffer
        buffer = io.BytesIO()
        sf.write(buffer, processed_y, sr, format='wav')
        buffer.seek(0)
        
        st.markdown("---")
        st.markdown("#### 🎧 Live Player & Download")
        
        # Output elements reflect changes instantly
        st.audio(buffer.getvalue(), format="audio/wav")
        st.download_button(
            label="📥 Download Processed Audio",
            data=buffer.getvalue(),
            file_name="CA_Editor_Output.wav",
            mime="audio/wav"
    )
        
