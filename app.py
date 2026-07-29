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
st.write("Upload any audio file below to instantly modify its speed, change its pitch, or boost the bass. All processing is done securely in memory and never stored on our servers.")

# 2. Upload Box
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Load audio array into memory using librosa
    st.info("Loading audio file... Please wait.")
    y, sr = librosa.load(uploaded_file, sr=None)
    st.success("Audio loaded successfully!")
    
    st.markdown("---")
    st.markdown("#### 🎛️ Audio Controls")
    
    # 3. Dynamic User Interface Sliders
    speed_rate = st.slider("🏃 Speed Modifier", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="0.5x is slow motion, 2.0x is fast forward.")
    pitch_steps = st.slider("🗣️ Pitch Changer (Semitones)", min_value=-8, max_value=8, value=0, step=1, help="Negative values make the voice deeper; positive values make it higher.")
    bass_boost_db = st.slider("🔊 Bass Booster (dB)", min_value=0, max_value=15, value=0, step=1, help="Increase to pump up the low-end bass frequencies.")
    
    # Process audio button to avoid lag while adjusting sliders
    if st.button("✨ Apply Effects"):
        with st.spinner("Processing your audio custom effects..."):
            processed_y = y.copy()
            
            # A. Apply Pitch Change
            if pitch_steps != 0:
                processed_y = librosa.effects.pitch_shift(processed_y, sr=sr, n_steps=pitch_steps)
                
            # B. Apply Speed Change
            if speed_rate != 1.0:
                processed_y = librosa.effects.time_stretch(processed_y, rate=speed_rate)
                
            # C. Apply Bass Boost (Low-shelf Filter logic via Scipy)
            if bass_boost_db > 0:
                # Basic low-pass filter logic to isolate and boost low frequencies (< 200Hz)
                f0 = 200.0  # Cutoff frequency for bass
                w0 = f0 / (sr / 2)
                gain = 10 ** (bass_boost_db / 20)
                
                # Setup simple first-order low shelf filter coefficients
                b = [1, w0 * (gain - 1)]
                a = [1, w0 * (1/gain - 1)]
                
                # Apply filter to the audio array
                processed_y = lfilter(b, a, processed_y)
                
                # Normalize audio to prevent clipping/distortion from the boost
                max_val = np.max(np.abs(processed_y))
                if max_val > 0:
                    processed_y = processed_y / max_val

            # 4. Convert processed sound matrix back to a file buffer
            buffer = io.BytesIO()
            sf.write(buffer, processed_y, sr, format='wav')
            buffer.seek(0)
            
            st.markdown("---")
            st.markdown("#### 🎧 Your Output File")
            
            # Show preview player and download options
            st.audio(buffer.getvalue(), format="audio/wav")
            st.download_button(
                label="📥 Download Processed Audio",
                data=buffer.getvalue(),
                file_name="CA_Editor_Output.wav",
                mime="audio/wav"
                                                         )
                
