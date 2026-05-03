# The Hotel VOC Analysis

Voice of customer analysis on 47,741 bilingual hotel reviews using XLM-RoBERTa sentiment classification and BERTopic topic modelling to identify the top operational areas driving guest dissatisfaction.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://github.com/huggingface/transformers)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.16+-green.svg)](https://bertopic.com/)

---

## Business Value

THEHOTEL.com management cannot act on tens of thousands of individual reviews. This analysis answers three operational questions:

1. Which areas generate the most dissatisfaction across brands?
2. Which guest segments are most dissatisfied?
3. Are complaint themes shifting over time?

**Key findings:**
- Room quality and service consistency account for 55.9% of all negative reviews and are stable across the two-month window — a structural problem, not a transient event
- Business travellers are the most dissatisfied segment (35.5% negative rate), followed by the 25–34 age group (34%)
- Brand 36 is a clear outlier with a 64% negative rate on 25 reviews; brands 2 and 0 (Ibis) show a distinct price-quality mismatch complaint cluster

**Recommended actions (priority order):**
1. Set group-wide minimum service standards for business traveller segments
2. Investigate room and service complaints at brands 35, 28, 33, and 31 with a shared root cause analysis
3. Audit brand 36 directly — small volume, extreme negative rate
4. Review Ibis room quality and pricing communication for brands 2 and 0

---

## Project Structure

```
thehotel-voc-analysis/
├── notebooks/
│   └── thehotel_analysis.ipynb   # full pipeline: cleaning, EDA, sentiment, BERTopic
├── figures/
│   ├── fig_sentiment_trend.png
│   ├── fig_neg_by_segment.png
│   └── fig_brand_concentration.png
├── voc_analysis_report.pdf
├── requirements.txt
└── README.md
```

**Note:** `reviews.csv` is not included (proprietary to course).

---

## Quick Start

<details>
<summary><b>Setup and Execution — Click to expand</b></summary>

### Prerequisites
- Python 3.10+
- Google Colab recommended (GPU required for XLM-RoBERTa inference, ~10 min on T4)

### Setup

1. **Clone repository**
```bash
git clone https://github.com/wengchienwei/thehotel-voc-analysis.git
cd thehotel-voc-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run notebook**

Upload `reviews.csv` to `/content/` in Colab, then run `thehotel_analysis.ipynb` top to bottom.

```
Note: Dataset not included. If you have access to the reviews.csv file:
1. Upload it to /content/reviews.csv in Colab
2. Run the notebook sequentially from Part 1
```

</details>

---

## Methodology

**Pipeline:**
1. **Cleaning** — whitespace stripping, categorical normalisation, deduplication on hotel code + date + text; missing demographics retained as `not disclosed`
2. **EDA** — review volume trend, demographic distributions, language split by brand, cross-tabulations
3. **Sentiment classification** — XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) as primary model; handles English and French natively. VADER applied to English-only reviews for methodological comparison
4. **Topic modelling** — BERTopic on high-confidence negative reviews (score ≥ 0.745); multilingual embeddings via `paraphrase-multilingual-MiniLM-L12-v2`; combined English–French stop-word list passed to CountVectorizer

**Why XLM-RoBERTa over VADER as primary?** The dataset is 40% French. VADER's lexicon is English-only and produces near-neutral scores on French text. Agreement between the two models on English reviews is 78.8%, with VADER over-classifying positive sentiment in mixed reviews.

---

## Tech Stack

**Core:** Python 3.10, pandas, numpy, matplotlib

**NLP:** transformers, torch, datasets, vaderSentiment, nltk

**Topic modelling:** bertopic, sentence-transformers, hdbscan, scikit-learn

---

## License

Code: MIT License

Dataset: Not included (proprietary to course)

---

## Author

Chien-Wei Weng

MSc Data Sciences and Business Analytics

CentraleSupélec × ESSEC Business School

---

**Academic Project | AI in Business & Data Monetization (Spring 2026)**
