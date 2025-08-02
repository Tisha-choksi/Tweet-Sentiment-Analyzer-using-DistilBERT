
# Tweet Sentiment Analyzer 🐦

A Streamlit web application for analyzing sentiment in tweets using a fine-tuned DistilBERT model trained on the tweet_eval dataset.

## Features

- **Sentiment Analysis**: Classifies tweets as Positive, Negative, or Neutral  
- **Confidence Scores**: Shows model confidence for each prediction  
- **Batch Processing**: Analyze multiple tweets from a text file  
- **Visual Dashboard**: Interactive charts and summary statistics  
- **Responsive Design**: Works on both desktop and mobile devices  

## Technical Stack

- 🤗 Transformers (DistilBERT model)  
- 🚀 Streamlit for web interface  
- 🐍 Python 3.8+  
- 🏗️ Hugging Face ecosystem  

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tweet-sentiment-analyzer.git
   cd tweet-sentiment-analyzer
  

2. Install dependencies:
  ```bash
   pip install -r requirements.txt
   

3. Download the pre-trained model (or train your own):
  ```bash
 If you have trained your own model, ensure it's in ./final_sentiment_model



```
## Usage

### Running Locally

```bash
streamlit run app.py
```

### Training the Model

The training notebook (`train_model.ipynb`) contains the complete workflow:

1. Loads the tweet\_eval dataset
2. Fine-tunes DistilBERT
3. Evaluates performance
4. Saves the model

### Deployment Options

1. **Streamlit Sharing**

   * Push to GitHub
   * Connect at [share.streamlit.io](https://share.streamlit.io/)

2. **Hugging Face Spaces**

   * Create new Space
   * Upload all files including `requirements.txt`

## File Structure

```
tweet-sentiment-analyzer/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── final_sentiment_model/  # Saved model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── train_model.ipynb       # Training notebook
└── README.md               # This file
```

## Model Performance

The fine-tuned model achieves:

* Accuracy: \~70% on test set
* Class-wise performance:

  * Positive: 88% recall
  * Negative: 66% recall
  * Neutral: 67% recall

## Customization

To modify the application:

1. Change model parameters in `app.py`
2. Adjust training settings in the notebook
3. Update the UI in the Streamlit components

## Troubleshooting

### Common Issues

1. **Model not loading**

   * Verify all model files are present in `final_sentiment_model/`
   * Check file permissions

2. **CUDA errors**

   * Set `device=-1` in `pipeline()` for CPU-only operation
   * Install correct PyTorch version for your GPU

3. **Dependency conflicts**

   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Contributing

Pull requests are welcome! For major changes, please open an issue first.


