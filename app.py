import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import os

# App configuration
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stTextInput input, .stTextArea textarea {
        font-size: 16px !important;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .negative {
        color: #ff4b4b;
    }
    .positive {
        color: #4CAF50;
    }
    .neutral {
        color: #1DA1F2;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_path = "./final_sentiment_model"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            st.error("Model files not found! Please make sure you've saved your model.")
            return None
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Main app function
def main():
    st.title("🐦 Tweet Sentiment Analyzer")
    st.markdown("Analyze sentiment using your fine-tuned DistilBERT model")
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        analyze_option = st.radio(
            "Analysis Mode:",
            ("Single Text", "Batch File")
        )
        
        st.markdown("---")
        st.markdown("""
        **Model Info:**
        - DistilBERT base uncased
        - Fine-tuned on tweet_eval
        - 3 classes: Positive/Negative/Neutral
        """)
    
    # Main content area
    if analyze_option == "Single Text":
        st.subheader("Analyze Single Tweet")
        text_input = st.text_area(
            "Enter your text here:",
            "I really enjoyed the movie! The acting was great.",
            height=100
        )
        
        if st.button("Analyze", key="analyze_single"):
            if not text_input.strip():
                st.warning("Please enter some text to analyze")
            else:
                with st.spinner("Processing..."):
                    classifier = load_model()
                    if classifier:
                        try:
                            result = classifier(text_input)[0]
                            display_result(result)
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
    
    else:  # Batch mode
        st.subheader("Batch Analysis")
        uploaded_file = st.file_uploader(
            "Upload a text file (one tweet per line)",
            type=["txt", "csv"]
        )
        
        if uploaded_file is not None:
            if st.button("Analyze", key="analyze_batch"):
                with st.spinner("Processing batch file..."):
                    classifier = load_model()
                    if classifier:
                        try:
                            lines = uploaded_file.getvalue().decode("utf-8").splitlines()
                            results = []
                            
                            # Process first 100 lines max (for demo)
                            for line in lines[:100]:
                                if line.strip():
                                    result = classifier(line.strip())[0]
                                    results.append({
                                        "text": line[:50] + "..." if len(line) > 50 else line,
                                        "sentiment": result['label'],
                                        "confidence": f"{result['score']:.2f}"
                                    })
                            
                            # Show results
                            st.dataframe(
                                results,
                                use_container_width=True,
                                column_config={
                                    "text": "Tweet",
                                    "sentiment": st.column_config.TextColumn(
                                        "Sentiment",
                                        help="Predicted sentiment"
                                    ),
                                    "confidence": st.column_config.ProgressColumn(
                                        "Confidence",
                                        help="Model confidence",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1
                                    )
                                }
                            )
                            
                            # Show summary
                            show_summary(results)
                            
                        except Exception as e:
                            st.error(f"Batch processing failed: {str(e)}")

# Helper function to display single result
def display_result(result):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Predicted Sentiment", result['label'])
    
    with col2:
        confidence = result['score']
        st.write(f"Confidence: {confidence:.2%}")
        st.progress(confidence)
    
    # Visual indicator
    sentiment = result['label'].lower()
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
        <h4 style="color: #333; margin-bottom: 5px;">Analysis Result:</h4>
        <p style="font-size: 18px;">
            This text is <span class="{sentiment}">{result['label']}</span> 
            with {confidence:.2%} confidence
        </p>
    </div>
    """, unsafe_allow_html=True)

# Helper function for batch summary
def show_summary(results):
    if not results:
        return
    
    counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }
    
    for r in results:
        counts[r['sentiment']] += 1
    
    total = len(results)
    st.subheader("Batch Summary")
    
    cols = st.columns(3)
    colors = ["#4CAF50", "#1DA1F2", "#FF4B4B"]
    
    for (sentiment, count), color in zip(counts.items(), colors):
        with cols[list(counts.keys()).index(sentiment)]:
            st.metric(
                label=sentiment,
                value=f"{count} ({count/total:.1%})",
                help=f"Number of {sentiment} tweets"
            )
    
    # Pie chart
    chart_data = {
        "Sentiment": list(counts.keys()),
        "Count": list(counts.values())
    }
    st.bar_chart(chart_data, x="Sentiment", y="Count", color=colors)

if __name__ == "__main__":
    main()