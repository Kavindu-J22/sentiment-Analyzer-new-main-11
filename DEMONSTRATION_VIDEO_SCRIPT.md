# 🎥 Ultimate Demonstration Video Script & Guide
## Project: Quantifying Customer Dissatisfaction — A Hybrid NLP Approach
### Target Duration: 20 Minutes

This document provides a highly comprehensive script for your 20-minute research project demonstration, with an expanded focus on the Streamlit dashboard functionality.

---

## ⏱️ Video Timeline Overview

| Section | Duration | Key Focus |
| :--- | :--- | :--- |
| **1. Introduction & Research Context** | 4 Minutes | Problem Statement, Objectives, and Repo Overview |
| **2. Code & Logic Walkthrough** | 6 Minutes | Phase I (Preprocessing), Phase II (NLP Engines), src/ folder |
| **3. Deep-Dive Dashboard Demo** | 10 Minutes | Comprehensive walkthrough of all 7 pages & features |
| **4. Conclusion & Research Findings** | 3 Minutes | Key Insights, Contributions, and Future Work |

---

## 🎤 Part 1: Introduction & Research Context (0:00 - 4:00)

### 🎬 Action: Show your GitHub Repository / Project Folder
**Say this:**
"Hello everyone. My name is **[Your Name]**, and today I am presenting my final year research project: **'Quantifying Customer Dissatisfaction: A Hybrid NLP Approach to Analysing Online Reviews'**.

### 1.1 The Problem Statement
In today's e-commerce landscape, customer feedback is the most valuable asset a company has. However, most businesses still rely on binary sentiment analysis—simply labeling a review as 'positive' or 'negative'. This is fundamentally flawed because:
- **Star ratings are misleading**: A customer might give 4 stars but express a specific complaint in the text that gets ignored.
- **Negativity is nuanced**: There is a huge difference between a customer who is 'mildly annoyed' and one who is 'extremely angry'.
- **Root causes are hidden**: Knowing a customer is unhappy doesn't tell a business *why*. Is it the fit? the fabric? or the delivery?

### 1.2 Research Objectives
My project addresses these gaps with four primary objectives:
1. To develop a **continuous 0–100 Dissatisfaction Index** based on the VADER lexicon.
2. To implement **Unsupervised Topic Modeling (LDA)** to automatically categorize complaints into business domains.
3. To build a **Sarcasm Detection layer** using Transformer models (RoBERTa) to catch negative reviews hidden behind positive words.
4. To integrate these into a **Real-time Analytics Dashboard** for business stakeholders.

---

## 💻 Part 2: Code & Logic Walkthrough (4:00 - 10:00)

### 🎬 Action: Open VS Code and show the `src/` folder
"Let's dive into the technical implementation. My system follows a **3-Phase Architecture**."

#### Phase I: Preprocessing (`src/preprocessing.py`)
- **Function: `clean_text()` (Lines 66–84)**: Note **Line 82** where I use regex `[^a-z\s]` to keep only letters.
- **Function: `replace_slang()` (Lines 59–63)**: Converts terms like 'tts' to 'true to size' (Lines 25–40).
- **Function: `lemmatize_text()` (Lines 87–98)**: We use POS-aware lemmatization. On **Line 46**, I explicitly **preserve negation words** like 'not' and 'never'.

#### Phase II: The 3 NLP Engines
1. **`sentiment_vader.py`**:
   - **`analyze_single()` (Lines 44–60)**: Computes the **Dissatisfaction Score** on **Line 52**: `round(max(0.0, -compound) * 100, 2)`.
2. **`topic_modeling.py`**:
   - **`train_lda_model()` (Lines 66–81)**: Trains a Gensim LDA model with `num_topics=6`.
   - **`get_coherence_score()` (Lines 84–91)**: Calculates the **C_v score** for mathematical proof of topic quality.
3. **`sarcasm_detector.py`**:
   - **`_lexical_sarcasm_boost()` (Lines 44–88)**: Detects 'sarcasm patterns' like quoted praise.
   - **`detect_single()` (Lines 146–191)**: Combines RoBERTa with the lexical layer for the final `irony_prob`.

---

## 📊 Part 3: Deep-Dive Dashboard Demo (10:00 - 20:00)

### 🎬 Action: Open the browser at `http://localhost:8501`

#### 3.1 Page 1: 🏠 Home & Overview (1.5 mins)
- **Feature Walkthrough**:
    - "The **Overview Metric Cards** give an immediate snapshot: 23,486 reviews and 3 NLP engines."
    - "The **Methodology Tab** is crucial. It explains that VADER measures *Intensity*, LDA measures *Theme*, and RoBERTa measures *Nuance*."
    - "The **Research Phases Tab** shows our visual pipeline: Data Hub → Hybrid Engine → Analytics Dashboard."
    - "This page acts as the 'Control Center' for the entire demonstration."

#### 3.2 Page 2: 📂 Data Hub (2 mins)
- **Feature Walkthrough**:
    - "We have three input modes. I've designed the **Upload CSV/Excel** feature (Lines 347–362) so businesses can analyze their own data instantly."
    - **Live Analysis Interaction**: "Let's use the **Live Review Text** mode. I'll enter: *'Wow, what a "high quality" dress. It literally fell apart the first time I wore it.'*"
    - "Note the **RoBERTa Status Indicator**. I'll click 'Load RoBERTa' (Line 406). This downloads the 500MB transformer model. Once loaded, click 'Analyse'."
    - **Explaining Results**: "Look at the **Sarcasm Correction** (Lines 617–627). VADER was fooled by 'high quality' and predicted 'Positive'. But RoBERTa identified a 92% irony probability, and my system **overrode the sentiment to Negative (Sarcasm 🎭)**. This is the hybrid system's greatest strength."

#### 3.3 Page 3: 🔧 Phase I – Preprocessing (1 min)
- **Feature Walkthrough**:
    - "Click 'Run Preprocessing' (Line 717). This cleans the 23k reviews in real-time."
    - "The **Cleaned Data Tab** shows the raw text next to the lemmatized `processed_text`."
    - "The **Feature Stats Tab** shows box plots (Lines 752–756) for word counts. Angry customers write longer reviews—this chart confirms that theory."
    - "The **Slang Examples Tab** lists all the fashion-domain abbreviations we normalize."

#### 3.4 Page 4: 📊 Phase II – Sentiment (VADER) (1.5 mins)
- **Feature Walkthrough**:
    - "Run the VADER Analysis (Line 821). The **Dissatisfaction Index Gauge** (Lines 845–847) shows our brand health score."
    - "The **Scatter Plot** (Lines 858–859) is vital. It plots VADER Compound against Star Rating. The clear diagonal trend validates that our machine score aligns with human judgment."
    - "The **Review Explorer** (Lines 876–884) allows us to filter by 'Severely Dissatisfied'. We can read the exact texts that scored 90/100."

#### 3.5 Page 5: 🗂️ Phase II – Topic Modeling (LDA) (1.5 mins)
- **Feature Walkthrough**:
    - "Here we train our **LDA Model**. I've added sliders (Lines 913–914) to adjust the number of topics and training passes."
    - "The **Topic Keyword Cards** (Lines 950–958) show the probability weights for words like 'fit', 'fabric', and 'delivery'."
    - "The **Heatmap** (Lines 960–969) is the most actionable chart. Dark red cells identify the specific intersections—like 'Product Quality' at 1-star—that require immediate management attention."

#### 3.6 Page 6: 🎭 Phase II – Sarcasm (RoBERTa) (1 min)
- **Feature Walkthrough**:
    - "We can analyze a sample of up to 3,000 reviews for sarcasm (Lines 1022–1024)."
    - "The **Donut Chart** (Lines 1086–1087) reveals the sarcasm rate in our dataset."
    - "The **Disagreement Analysis** (Lines 1103–1111) is the 'Smoking Gun'. It lists all reviews that VADER thought were happy but RoBERTa correctly identified as unhappy."

#### 3.7 Page 7: 📈 Phase III – Analytics Dashboard (1.5 mins)
- **Feature Walkthrough**:
    - "The final page is an **Integrated Executive Summary**."
    - "The **Word Clouds** (Lines 1196–1223) show the difference between positive praise and negative complaints at a glance."
    - "The **Age & Department Analysis** (Lines 1184–1191) helps identify if younger or older customers are more unhappy, or if specific departments like 'Dresses' have a quality issue."
    - "Finally, the **Export Section** (Lines 1268–1289) allows stakeholders to download all analyzed data, including irony probabilities, for use in their own business intelligence tools."

---

## 🏁 Part 4: Conclusion & Research Findings (17:00 - 20:00)

### 🎬 Action: Show the Heatmap or the Analytics Dashboard again
**Say this:**
"To conclude my demonstration, I'd like to summarize the key findings and contributions of this research.

### 4.1 Key Findings
1. **The Sarcasm Blind Spot**: My research found that standard sentiment tools misclassify roughly **3.5% of reviews** due to sarcasm. By using a hybrid RoBERTa + Lexical approach, we recovered these 'lost' negative signals.
2. **Dissatisfaction vs. Rating**: We confirmed that star ratings are often 'sticky'—customers give 3 or 4 stars but express 80/100 dissatisfaction in the text. Our NLP model is more sensitive to these operational issues than the rating column.
3. **Topic Severity**: Unsupervised modeling revealed that **Product Quality** and **Fit/Size** are the primary drivers of severe dissatisfaction in this domain, accounting for over 60% of high-dissatisfaction scores.

### 4.2 Research Contributions
This project contributes three novel elements:
- **A Continuous Metric**: Moving from binary labels to a 0–100 scale for business-level quantification.
- **Hybrid Sarcasm Architecture**: Combining transformer deep learning with domain-specific lexical heuristics.
- **Operational Heatmaps**: Cross-referencing unsupervised topics with dissatisfaction scores to create a visual 'pain-point' map.

### 4.3 Limitations and Future Work
While successful, there are limitations:
- **Domain Specificity**: The current model is tuned for e-commerce. Future work could adapt this to the hospitality or healthcare sectors.
- **ABSA Integration**: The next step would be **Aspect-Based Sentiment Analysis**, allowing us to see specifically that 'Zippers' or 'Buttons' are the issue within the Quality topic.

This project demonstrates that quantifying customer dissatisfaction through a hybrid NLP approach provides deeper, more actionable intelligence than traditional methods. Thank you for your time."
