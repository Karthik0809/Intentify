import streamlit as st
import json
import random
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from app.nlp_module import IntentClassifier, classify_intent, load_model_and_encoder
from app_config import MODEL_PATH, INTENTS
from transformers import BertTokenizer
import pandas as pd
from datetime import datetime
import time
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment
import io

# 🎛 Page setup
st.set_page_config(page_title="Intentify", page_icon="🎯", layout="wide")
st.title("🎯 Intentify - Smart Intent Classification")
st.write("An advanced intent classification system powered by BERT and Snips dataset, capable of understanding both text and voice inputs with high accuracy.")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []
if "text" not in st.session_state:
    st.session_state.text = ""
if "intent" not in st.session_state:
    st.session_state.intent = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "error" not in st.session_state:
    st.session_state.error = None
if "show_visualizations" not in st.session_state:
    st.session_state.show_visualizations = True
if "show_history" not in st.session_state:
    st.session_state.show_history = True

# 🧠 Load trained model and tokenizer
@st.cache_resource
def get_model_and_tokenizer():
    try:
        model, label_encoder = load_model_and_encoder()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, tokenizer, label_encoder = get_model_and_tokenizer()

# Update session state with label encoder
if label_encoder is not None:
    st.session_state.label_encoder = label_encoder

# 📂 Load data from JSON file
@st.cache_data
def load_examples():
    try:
        with open("data/snips_augmented.json", "r") as f:
            return json.load(f)["data"]
    except Exception as e:
        st.error(f"Error loading examples: {str(e)}")
        return []

examples = load_examples()

# Define supported intents
intents = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetNews",
    "GetWeather",
    "Other",
    "PlayMusic",
    "RateBook",
    "SearchScreeningEvent",
    "SetTimer"
]

def transcribe_audio(audio_file):
    """Transcribe audio file to text using speech recognition"""
    try:
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Convert audio file to WAV if it's not already
        if audio_file.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_file)
            wav_file = audio_file.replace('.mp3', '.wav')
            audio.export(wav_file, format="wav")
            audio_file = wav_file
        
        # Load audio file
        with sr.AudioFile(audio_file) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            # Record audio
            audio = r.record(source)
            
        # Transcribe audio
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Create main tabs for all sections
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "🎙️ Voice Input", 
    "📝 Text Input", 
    "🎲 Random Samples", 
    "📊 Quick Stats",
    "ℹ️ Model Info"
])

with main_tab1:
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #f5f5f5; margin: 10px 0;'>
            <h3 style='color: #1E88E5; margin: 0 0 10px 0;'>Voice Input</h3>
            <p style='color: #666; margin: 0;'>Click the button below and speak to classify your intent</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎙️ Start Recording", use_container_width=True):
        try:
            # Initialize recognizer
            r = sr.Recognizer()
            
            # Use microphone as source
            with sr.Microphone() as source:
                st.markdown("""
                    <div style='padding: 15px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                        <p style='color: #1E88E5; margin: 0;'>🎤 Listening... Speak now!</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source)
                # Record audio
                audio = r.listen(source)
            
            with st.spinner("Processing audio..."):
                # Transcribe audio to text
                transcribed_text = r.recognize_google(audio)
                
                if transcribed_text:
                    st.markdown(f"""
                        <div style='padding: 15px; border-radius: 10px; background-color: #f5f5f5; margin: 10px 0;'>
                            <p style='color: #666; margin: 0;'>Transcribed text: <strong>{transcribed_text}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Classify the transcribed text
                    try:
                        with st.spinner("Analyzing intent..."):
                            intent, confidence = classify_intent(transcribed_text, model, return_confidence=True)
                            
                            # Update session state
                            st.session_state.text = transcribed_text
                            st.session_state.intent = intent
                            st.session_state.confidence = confidence
                            
                            # Add to history
                            st.session_state.history.append({
                                "text": transcribed_text,
                                "intent": intent,
                                "confidence": confidence,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Display results with custom styling
                            st.markdown(f"""
                                <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                                    <h3 style='color: #1E88E5; margin: 0;'>Detected Intent: {intent}</h3>
                                    <p style='color: #666; margin: 10px 0 0 0;'>Confidence: {confidence:.2%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
                        st.session_state.error = str(e)
        except sr.UnknownValueError:
            st.error("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except Exception as e:
            st.error(f"Error processing audio: {e}")

with main_tab2:
    st.markdown('<div class="main-header">📝 Text Input</div>', unsafe_allow_html=True)
    
    # Text input with modern styling
    st.markdown("""
        <style>
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 15px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Enter your text:",
        value=st.session_state.text,
        height=100,
        help="Enter text to classify its intent"
    )
    
    # Classification button with custom styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #1E88E5;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Classify Text", use_container_width=True):
        if not model or not tokenizer:
            st.error("Model not loaded properly. Please check the model files.")
        elif not user_input:
            st.warning("Please enter some text to classify.")
        else:
            try:
                with st.spinner("Analyzing intent..."):
                    # Get intent and confidence
                    intent, confidence = classify_intent(user_input, model, return_confidence=True)
                    
                    # Update session state
                    st.session_state.text = user_input
                    st.session_state.intent = intent
                    st.session_state.confidence = confidence
                    
                    # Add to history
                    st.session_state.history.append({
                        "text": user_input,
                        "intent": intent,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results with custom styling
                    st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                            <h3 style='color: #1E88E5; margin: 0;'>Detected Intent: {intent}</h3>
                            <p style='color: #666; margin: 10px 0 0 0;'>Confidence: {confidence:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")
                st.session_state.error = str(e)

with main_tab3:
    st.markdown('<div class="main-header">🎲 Random Samples</div>', unsafe_allow_html=True)
    
    if examples:
        if st.button("🎲 Try Random Example", use_container_width=True):
            sample = random.choice(examples)
            st.session_state.text = sample["text"]
            
            st.markdown(f"""
                <div style='padding: 15px; border-radius: 10px; background-color: #f5f5f5; margin: 10px 0;'>
                    <p style='color: #666; margin: 0;'>Sample Text: <strong>{sample['text']}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                with st.spinner("Analyzing intent..."):
                    intent, confidence = classify_intent(sample["text"], model, return_confidence=True)
                    st.session_state.intent = intent
                    st.session_state.confidence = confidence
                    
                    # Add to history
                    st.session_state.history.append({
                        "text": sample["text"],
                        "intent": intent,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results with custom styling
                    st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                            <h3 style='color: #1E88E5; margin: 0;'>Detected Intent: {intent}</h3>
                            <p style='color: #666; margin: 10px 0 0 0;'>Confidence: {confidence:.2%}</p>
                            <p style='color: #666; margin: 10px 0 0 0;'>True Intent: {sample['intent']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

with main_tab4:
    st.markdown('<div class="main-header">📊 Quick Stats</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # Calculate statistics
        total_predictions = len(st.session_state.history)
        avg_confidence = np.mean([h["confidence"] for h in st.session_state.history])
        intent_counts = pd.Series([h["intent"] for h in st.session_state.history]).value_counts()
        
        # Display statistics in a modern card layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                    <h3 style='color: #1E88E5; margin: 0;'>Total Predictions</h3>
                    <p style='color: #666; margin: 10px 0 0 0; font-size: 24px;'>{total_predictions}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
                    <h3 style='color: #1E88E5; margin: 0;'>Average Confidence</h3>
                    <p style='color: #666; margin: 10px 0 0 0; font-size: 24px;'>{avg_confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Add visualization toggle
        st.session_state.show_visualizations = st.checkbox("Show Visualizations", value=st.session_state.show_visualizations)
        
        # Plot intent distribution
        if st.session_state.show_visualizations:
            fig = px.pie(
                values=intent_counts.values,
                names=intent_counts.index,
                title="Intent Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

with main_tab5:
    st.markdown('<div class="main-header">ℹ️ Model Information</div>', unsafe_allow_html=True)
    
    # Model Architecture Section
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
            <h3 style='color: #1E88E5; margin: 0 0 15px 0;'>Model Architecture</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                    <h4 style='color: #1E88E5; margin: 0 0 10px 0;'>Base Model</h4>
                    <ul style='color: #666; margin: 0; padding-left: 20px;'>
                        <li>BERT-based architecture</li>
                        <li>Pre-trained on large text corpus</li>
                        <li>Fine-tuned for intent classification</li>
                    </ul>
                </div>
                <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                    <h4 style='color: #1E88E5; margin: 0 0 10px 0;'>Training Process</h4>
                    <ul style='color: #666; margin: 0; padding-left: 20px;'>
                        <li>5-fold cross-validation</li>
                        <li>Enhanced data augmentation</li>
                        <li>Optimized hyperparameters</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Performance Section
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
            <h3 style='color: #1E88E5; margin: 0 0 15px 0;'>Model Performance</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
                <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                    <h4 style='color: #1E88E5; margin: 0 0 10px 0;'>Accuracy</h4>
                    <p style='color: #666; margin: 0; font-size: 24px;'>100%</p>
                    <p style='color: #666; margin: 5px 0 0 0; font-size: 14px;'>on validation set</p>
                </div>
                <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                    <h4 style='color: #1E88E5; margin: 0 0 10px 0;'>Cross-Validation</h4>
                    <p style='color: #666; margin: 0; font-size: 24px;'>5-fold</p>
                    <p style='color: #666; margin: 5px 0 0 0; font-size: 14px;'>robust evaluation</p>
                </div>
                <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                    <h4 style='color: #1E88E5; margin: 0 0 10px 0;'>Data Augmentation</h4>
                    <p style='color: #666; margin: 0; font-size: 24px;'>Enhanced</p>
                    <p style='color: #666; margin: 5px 0 0 0; font-size: 14px;'>for better generalization</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Supported Intents Section
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
            <h3 style='color: #1E88E5; margin: 0 0 15px 0;'>Supported Intents</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>
    """, unsafe_allow_html=True)
    
    for intent in intents:
        st.markdown(f"""
            <div style='padding: 12px; background-color: #f5f5f5; border-radius: 8px;'>
                <p style='color: #666; margin: 0; font-weight: 500;'>{intent}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #e3f2fd; margin: 10px 0;'>
            <h3 style='color: #1E88E5; margin: 0 0 15px 0;'>Dataset Information</h3>
            <div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>
                <p style='color: #666; margin: 0;'>The model is trained on the Snips dataset, which contains a diverse set of user queries across multiple domains. The dataset has been augmented to improve model robustness and generalization capabilities.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# History and Visualizations Section
if st.session_state.show_history and st.session_state.history:
    st.markdown("---")
    st.header("📜 Prediction History")
    
    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.history)
    
    # Display history table
    st.dataframe(
        history_df.style.format({"confidence": "{:.2%}"}),
        use_container_width=True
    )
    
    if st.session_state.show_visualizations:
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Confidence Over Time", "Intent Distribution"])
        
        with tab1:
            # Plot confidence over time
            fig = px.line(
                history_df,
                x="timestamp",
                y="confidence",
                title="Confidence Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Plot intent distribution
            intent_counts = pd.Series([h["intent"] for h in st.session_state.history]).value_counts()
            fig = px.bar(
                intent_counts,
                title="Intent Distribution",
                labels={"value": "Count", "index": "Intent"}
            )
            st.plotly_chart(fig, use_container_width=True)

# Error handling section
if st.session_state.error:
    st.error(f"Last Error: {st.session_state.error}")
    if st.button("Clear Error"):
        st.session_state.error = None
