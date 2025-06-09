# Intentify - Smart Intent Classification System

Intentify is an advanced intent classification system powered by BERT and the Snips dataset. It provides both text and voice input capabilities for classifying user intents with high accuracy.

## 🌟 Features

- 🎙️ Voice input processing with real-time transcription
- 📝 Text input classification with confidence scores
- 🎲 Random sample testing from the Snips dataset
- 📊 Real-time statistics and visualizations
- 🤖 BERT-based model architecture
- 📈 High accuracy predictions (100% on validation set)
- 🔄 5-fold cross-validation for robust evaluation
- 📱 Modern, responsive UI with Streamlit

## 🎯 Supported Intents

The system can classify the following intents:
- AddToPlaylist
- BookRestaurant
- GetNews
- GetWeather
- Other
- PlayMusic
- RateBook
- SearchScreeningEvent
- SetTimer

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intentify.git
cd intentify
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main.py
```

## 📁 Project Structure

```
intentify/
├── main.py                  # Main Streamlit application
├── train.py                 # Model training script
├── paraphrase_snips.py      # Data augmentation script
├── app/
│   ├── nlp_module.py        # NLP and model handling
│   └── __init__.py
├── data/
│   └── snips_augmented.json # Training data
├── models/                  # Saved model files
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md               # Project documentation
```

## 🛠️ Dependencies

- Streamlit - Web application framework
- PyTorch - Deep learning framework
- Transformers - BERT model implementation
- Plotly - Interactive visualizations
- SpeechRecognition - Voice input processing
- Pydub - Audio file handling
- NumPy - Numerical computations
- Pandas - Data manipulation
- scikit-learn - Machine learning utilities

## 🤖 Model Information

### Architecture
- BERT-based model for intent classification
- Pre-trained on large text corpus
- Fine-tuned on Snips dataset

### Training Process
- 5-fold cross-validation
- Enhanced data augmentation techniques
- Optimized hyperparameters
- 100% accuracy on validation set

### Data Augmentation
- Synonym replacement
- Case variations
- Paraphrasing
- Additional training examples

## 📊 Performance Metrics

- Validation Accuracy: 100%
- Cross-validation: 5-fold
- Robust evaluation metrics
- Real-time confidence scores

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Karthik Mulugu ([@karthikmulugu](https://www.linkedin.com/in/karthikmulugu/))

## 🙏 Acknowledgments

- Snips dataset for training data
- Hugging Face for BERT implementation
- Streamlit for the web interface 