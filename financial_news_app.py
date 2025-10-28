import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
import altair as alt

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Define the 20 Labels
LABEL_MAP = {
    0: 'Analyst Update', 1: 'Fed | Central Banks', 2: 'Company | Product News',
    3: 'Treasuries | Corporate Debt', 4: 'Dividend', 5: 'Earnings',
    6: 'Energy | Oil', 7: 'Financials', 8: 'Currencies',
    9: 'General News | Opinion', 10: 'Gold | Metals | Materials', 11: 'IPO',
    12: 'Legal | Regulation', 13: 'M&A | Investments', 14: 'Macro',
    15: 'Markets', 16: 'Real Estate', 17: 'Retail | Leisure',
    18: 'Science | Tech', 19: 'Transport | Logistics'
}
# Reverse map for topic selection in Tab 2
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# Pre-defined examples for easy user input
EXAMPLE_INPUTS = {
    "Enter a custom tweet...": "",
    "1. Earnings (Tesla)": "Tesla stock surges after beating Q4 earnings expectations, delivering record profits.",
    "2. Analyst Update (Disney)": "Analysts downgrade Disney stock citing slowdown in streaming subscriber growth.",
    "3. Fed | Central Banks (Rates)": "Fed Chair Powell signals aggressive interest rate hikes to combat persistent inflation.",
    "4. M&A | Investments (Cisco)": "Cisco announces deal to acquire security software firm Splunk for $28 billion.",
    "5. Macro (Job Report)": "The jobs report shows strong employment figures, defying recession fears.",
    "6. Energy | Oil (Supply)": "Oil prices fall sharply as Saudi Arabia boosts production quotas.",
    "7. IPO (Reddit)": "Reddit eyes IPO in 2024, seeking $10 billion valuation."
}


@st.cache_resource
def load_assets():
    """Loads the model and vectorizer with Streamlit caching."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the training data for the interactive EDA tab
        train_df = pd.read_csv('train_data.csv')
        train_df['Topic'] = train_df['label'].map(LABEL_MAP)
        
        return vectorizer, model, train_df
    except FileNotFoundError as e:
        st.error(f"❌ Error: Required file not found: {e.filename}. Ensure all files are in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading assets: Please verify your model files are correct. Details: {e}")
        return None, None, None

vectorizer, model, train_df = load_assets()


# --- 2. DATA PREPARATION FOR INTERACTIVE EDA & KPIs ---

@st.cache_data
def get_eda_data(df):
    """Prepare DataFrame for Altair chart (Topic Counts) and calculate Imbalance Ratio."""
    if df is None:
        return pd.DataFrame(), 0.0
        
    eda_df = df['Topic'].value_counts().reset_index()
    eda_df.columns = ['Topic', 'Count']
    
    # Calculate Imbalance Ratio: Largest Class / Smallest Class
    max_count = eda_df['Count'].max()
    min_count = eda_df['Count'].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    return eda_df, imbalance_ratio

eda_df, imbalance_ratio = get_eda_data(train_df)

@st.cache_data
def get_feature_importance_data(_model, _vectorizer, class_index, n=10):
    """
    NOTE: This function is kept for completeness/future use but is no longer called 
    in the main Streamlit layout.
    """
    if _model is None or _vectorizer is None:
        return pd.DataFrame()

    feature_names = _vectorizer.get_feature_names_out()
    class_coef = _model.coef_[class_index] 
    coef_series = pd.Series(class_coef, index=feature_names)
    
    top_positive = coef_series.nlargest(n)
    top_negative = coef_series.nsmallest(n)
    
    df_pos = pd.DataFrame({'Feature': top_positive.index, 'Coefficient': top_positive.values, 'Type': 'Positive'})
    df_neg = pd.DataFrame({'Feature': top_negative.index, 'Coefficient': top_negative.values, 'Type': 'Negative'})
    
    df_neg['Coefficient'] = -df_neg['Coefficient'] 
    
    return pd.concat([df_pos, df_neg])


def clean_text(text):
    """Simplified text cleaning function for inference."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


# --- 3. STREAMLIT APP LAYOUT ---

# Apply a neutral wide layout for aesthetics
st.set_page_config(
    page_title="Twitter Financial News Topic Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NEW: Initialize session state for the dynamic topic selection (still used for Sample Viewer)
if 'selected_topic' not in st.session_state:
    st.session_state['selected_topic'] = 'Fed | Central Banks' # Default topic

# --- CUSTOM CSS INJECTION (Kept for styling) ---
st.markdown(
    """
    <style>
    /* Custom CSS for Black-Sky Blue Gradient */
    .stApp {
        /* Subtle Dark Gradient: Black to Very Dark Blue/Sky Blue tint, prioritizing readability */
        background: linear-gradient(to bottom right, #000000, #0A192F 60%, #102030); 
        background-attachment: fixed; /* Fixes background position on scroll */
    }
    
    /* Ensure all text is readable on the dark background */
    body, h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }
    
    /* Sidebar Background - slightly lighter black for distinction */
    .st-emotion-cache-12fm5p4 { 
        background-color: #1A1A1A; 
    }

    /* Container/Card Backgrounds (for KPIs) - fixed dark color for contrast on gradient */
    [data-testid="stContainer"] {
        background-color: #1A1A1A; /* Darker card background */
        color: white; /* Ensure text inside is white */
        border: 1px solid #333333; /* Light border for separation */
    }

    /* FIX: Adjust metric value font size to ensure all text, including 'Linear (Low)', fits neatly */
    [data-testid="stMetricValue"] {
        color: #00FF7F !important; /* Bright green for financial numbers */
        font-size: 20px !important; /* Reduced font size for better fit in tight columns */
    }

    /* Metric Label (Title) */
    [data-testid="stMetricLabel"] > div {
        color: #B0C4DE !important; /* Light blue/gray for contrast */
    }

    /* Tabs Bar Background - ensure tab content is readable */
    .st-emotion-cache-1ftrz11 {
        background-color: #1A1A1A; 
    }
    
    /* Dataframe background */
    .stDataFrame {
        background-color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- END CUSTOM CSS INJECTION ---

# Attractive Title with Divider
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>💸 Financial News Topic </h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: #808080;'>A Logistic Regression model for classifying financial tweets into 20 specific topics.</p>", unsafe_allow_html=True)


# --- SIDEBAR (Aesthetically Organized) ---
with st.sidebar:
    st.markdown("<h2 style='color: #1E90FF;'>📝 Input Tweet for Classification</h2>", unsafe_allow_html=True)    
    st.markdown("---")
    st.header("✨ Live Prediction Input")
    
    # Select box for examples
    example_choice = st.selectbox(
        "1. Select an example to pre-fill:",
        options=list(EXAMPLE_INPUTS.keys())
    )
    
    default_text = EXAMPLE_INPUTS[example_choice]

    # The main text area uses the dynamically set default_text
    input_text = st.text_area(
        "2. Or, enter your custom tweet here:",
        value=default_text, # Set initial value from the selection
        placeholder="e.g., 'Fed raises interest rates by 75 basis points to combat inflation.'",
        height=150
    )


# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🚀 Real-Time Classification", "📊 Interactive Data & Performance"])

with tab1:
    st.subheader("Analyze a Financial Tweet")

    if st.button("Analyze Tweet", type="primary") and model and vectorizer:
        if input_text:
            with st.spinner('Classifying tweet...'):
                
                # Inference Logic
                cleaned_text = clean_text(input_text)
                X_input = vectorizer.transform([cleaned_text])
                
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                
                prediction_label = LABEL_MAP.get(prediction, "UNKNOWN TOPIC")
                confidence = np.max(probabilities)
                
                # Update session state for Tab 2's Sample Viewer
                st.session_state['selected_topic'] = prediction_label
                
                top_5_indices = np.argsort(probabilities)[::-1][:5]
                top_5_topics = [
                    (LABEL_MAP.get(i), probabilities[i])
                    for i in top_5_indices
                ]

                # --- Attractive Output Display ---
                st.markdown("### Classification Result")
                
                col_main, col_prob = st.columns([1, 1])

                with col_main:
                    st.success(f"**TOPIC:** {prediction_label}", icon="🎯")
                    
                    # Highlight confidence percentage
                    st.markdown(f"The model is <span style='color: #00FF7F; font-weight: bold;'>{confidence*100:.1f}%</span> confident in this classification.", unsafe_allow_html=True)
                    
                    # Display confidence using a custom progress bar
                    st.progress(confidence)

                with col_prob:
                    st.markdown("### Top 5 Confidences")
                    # Use a clean dataframe for top predictions
                    top_df = pd.DataFrame(top_5_topics, columns=['Topic', 'Confidence'])
                    
                    # Convert confidence to percentage string for display
                    top_df['Confidence_Percent'] = top_df['Confidence'] * 100
                    
                    st.dataframe(top_df[['Topic', 'Confidence']].style.format({'Confidence': "{:.1%}"}), hide_index=True, use_container_width=True)

                
                # --- Classification Probability Graph (Interactive) ---
                st.markdown("---")
                st.markdown("### 📊 Confidence Breakdown")
                
                # Altair chart for Top 5 Probabilities
                chart = alt.Chart(top_df).mark_bar().encode(
                    x=alt.X('Confidence_Percent', title='Confidence (%)', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Topic', sort='-x', title='Topic'),
                    tooltip=[alt.Tooltip('Topic'), alt.Tooltip('Confidence', format='.1%')],
                    # Color the top prediction with the highlight color
                    color=alt.condition(
                        alt.datum.Topic == prediction_label,
                        alt.value('#1E90FF'), # Sky Blue Highlight
                        alt.value('#B0C4DE')  # Default Light Blue
                    ),
                    # Add text labels on the bars
                    text=alt.Text('Confidence', format='.1%')
                ).properties(
                    title="Top 5 Predicted Topics"
                ).interactive()

                st.altair_chart(chart, use_container_width=True)


        else:
            st.warning("👈 Please enter text in the sidebar to get a prediction.")

with tab2:
    st.markdown("<h2 style='color: #1E90FF;'>Model Performance & Data Insights</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # --- MODULAR KPI SECTION ---
    
    col_kpi_left, col_kpi_right = st.columns(2)

    # 1. MODEL PERFORMANCE CARD
    with col_kpi_left:
        with st.container(border=True): # Card structure
            st.markdown("<h4 style='color: #1E90FF;'>🤖 Model Performance Metrics</h4>", unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            
            # Validation Accuracy 
            col_a.metric(
                "Validation Accuracy", 
                "77.8%", 
                delta="Baseline",
                help="Accuracy measured on the unseen validation dataset. Represents the overall fraction of correct predictions."
            )
            
            # Weighted Avg F1-Score 
            col_b.metric(
                "Weighted Avg F1-Score", 
                "83.0%", 
                help="The F1-score averaged across all classes, weighted by topic support. This is the best performance metric for imbalanced data."
            )

            # Total Topics
            col_c.metric(
                "Total Topics", 
                "20", 
                delta_color="off",
                help="The total number of distinct financial topics the model is trained to classify."
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # 2. DATA INSIGHTS CARD
    with col_kpi_right:
        with st.container(border=True): # Card structure
            st.markdown("<h4 style='color: #6A5ACD;'>🧪 Data & Feature Context</h4>", unsafe_allow_html=True)
            col_d, col_e, col_f = st.columns(3)

            # Imbalance Ratio
            imbalance_display = f"{imbalance_ratio:.1f}:1" if imbalance_ratio > 0 else "N/A"
            col_d.metric(
                "Max/Min Class Ratio", 
                imbalance_display, 
                delta="High Imbalance", 
                delta_color="inverse", 
                help="Ratio of the largest class count to the smallest class count in the training data. A high ratio indicates severe class imbalance."
            )

            # Vocabulary Size
            vocab_size = len(vectorizer.vocabulary_) if vectorizer and hasattr(vectorizer, 'vocabulary_') else 0
            col_e.metric(
                "Vocabulary Size", 
                f"{vocab_size:,}", 
                delta="TF-IDF Features",
                help="The total number of unique features (words/n-grams) the TF-IDF vectorizer learned. This is the input dimensionality."
            )
            
            # Model Complexity (Fixed to fit the box)
            col_f.metric(
                "Model Type", 
                "Linear (Low)", 
                help="Logistic Regression is a linear and computationally inexpensive model, ideal for baseline performance and fast inference."
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE EXPANDER SECTION (Detailed Breakdown) ---
    st.markdown("<br>", unsafe_allow_html=True) 
    with st.expander("📝 Detailed Metric Breakdown: Why F1-Score is Key (Click to Expand)", expanded=False):
        st.markdown(
            f"""
            ### Analysis of Model Performance on Imbalanced Financial Data:
            
            - **Data Skew:** The training data suffers from severe class imbalance, indicated by the **Max/Min Class Ratio of {imbalance_display}**. This means the most common topic is many times more frequent than the rarest.
            - **Misleading Accuracy:** The **Validation Accuracy (77.8%)** is the overall percentage correct, but it can be artificially inflated by the model simply guessing the common topics correctly.
            - **Reliable F1-Score:** The **Weighted Avg F1-Score (83.0%)** is the critical metric. It harmonically averages Precision and Recall for *every* topic, providing a much truer picture of overall performance across all 20 topics.
            """
        )
    
    st.markdown("---")
    
    # --- Interactive EDA (Altair Chart) ---
    st.subheader("📈 Topic Distribution ")
    st.markdown("Use this interactive bar chart to see the **imbalance** in the data. **Hover** to see exact counts.")

    if not eda_df.empty:
        # Altair chart with professional colors
        chart = alt.Chart(eda_df).mark_bar().encode(
            y=alt.Y('Topic:N', sort=alt.EncodingSortField(field="Count", op="sum", order='descending'), title="Financial Topic"),
            x=alt.X('Count:Q', title="Number of Tweets"),
            tooltip=['Topic', 'Count'],
            color=alt.condition(
                alt.datum.Topic == eda_df.iloc[0]['Topic'], 
                alt.value('#1E90FF'), # Highlight Color
                alt.value('#B0C4DE')  # Default Color
            )
        ).properties(
            title='Interactive Topic Distribution'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Cannot display Topic Distribution. The training data could not be loaded.")
        
    st.markdown("---")
    
    # --- Feature Importance Chart REMOVED ---
    
    # --- Sample Data Viewer (Interactive, linked to live prediction) ---
    st.subheader("📄 Sample Tweet Viewer")
    st.markdown("Select a topic to view up to 5 examples of the raw data used for training. **(Defaults to the last predicted topic from the 'Real-Time Classification' tab.)**")
    
    # Use the sorted topics for the selector
    sorted_topics = eda_df['Topic'].tolist() if not eda_df.empty else list(LABEL_MAP.values())
    
    # Logic to set the default topic based on the latest classification (session state)
    default_index = sorted_topics.index(st.session_state['selected_topic']) if st.session_state['selected_topic'] in sorted_topics else 0

    selected_topic = st.selectbox(
        "Choose a Topic to see Sample Tweets:",
        sorted_topics,
        index=default_index,
        key='sample_viewer_select'
    )
    
    if train_df is not None:
        sample_data = train_df[train_df['Topic'] == selected_topic][['text']]
        if not sample_data.empty:
            # Randomly sample up to 5 examples
            sample_df = sample_data.sample(min(5, len(sample_data)))
            
            st.dataframe(
                sample_df.reset_index(drop=True), 
                column_config={
                    "text": st.column_config.TextColumn("Sample Tweet")
                }, 
                use_container_width=True
            )
        else:
            st.info(f"No samples found for the topic: {selected_topic}")
    else:
        st.warning("Data viewer unavailable: training data not loaded.")