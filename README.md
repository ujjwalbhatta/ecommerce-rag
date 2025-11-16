# RAG-Based Product Price Prediction System

**Masters Project - Information Retrieval and Storage**

A comprehensive Retrieval-Augmented Generation (RAG) system that predicts product prices using three different retrieval methods (FAISS, BM25, Hybrid) with LLM-powered explanations.

## 🎯 Project Overview

This system uses Amazon product data to predict prices by retrieving similar products and analyzing their characteristics. It combines modern embedding techniques with traditional information retrieval methods and adds an LLM reasoning layer for explainability.

### Key Features

✅ **Three Retrieval Methods:**
- **FAISS**: Dense vector similarity using sentence embeddings
- **BM25**: Sparse keyword-based retrieval (traditional IR)
- **Hybrid**: Combines FAISS + BM25 with optional reranking

✅ **LLM Reasoning:**
- Detailed explanations for each prediction
- Shows top similar products that influenced the price
- Quality assessment (Excellent/Good/Fair/Poor)
- Similarity and price reasoning analysis

✅ **Comprehensive Evaluation:**
- 5 metrics: MAE, RMSE, R², MAPE, Median AE
- Side-by-side comparison of all methods
- Detailed visualizations for each method

✅ **Rich Output:**
- 11 output files including visualizations, predictions, and explanations
- CSV files with top 3 similar products for each prediction
- Publication-ready plots and comprehensive summary report

---

## 📋 Requirements

```bash
pip install numpy pandas scikit-learn sentence-transformers rank-bm25 faiss-cpu matplotlib seaborn
```

### Dependencies:
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Evaluation metrics
- `sentence-transformers` - Text embeddings
- `rank-bm25` - BM25 retrieval
- `faiss-cpu` - Vector similarity search
- `matplotlib` & `seaborn` - Visualizations

---

## 🚀 Quick Start

### 1. Download Dataset

Get the Amazon product metadata from: [https://amazon-reviews-2023.github.io]

Download the `meta_Electronics.jsonl` file (or any category you prefer).

### 2. Configure the System

Open the script and edit the configuration section at the top:

```python
# ============================================================================
# SIMPLE CONFIGURATION - Change these values!
# ============================================================================

FILE_PATH = '/path/to/your/meta_Electronics.jsonl'  # ← Change this!
OUTPUT_DIR = './complete_rag_outputs'

# Data settings
SAMPLE_SIZE = 12000          # Number of products to load
MIN_PRICE = 5.0              # Minimum price filter
MAX_PRICE = 1000.0           # Maximum price filter
FILTER_CATEGORIES = ["Computers"]  # Categories to include

# Model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model

# Retrieval settings
K_SIMILAR = 15               # Number of similar products (FAISS/BM25)
K_HYBRID = 15                # Final k for Hybrid mode
ALPHA = 0.5                  # Hybrid weight (0.5 = equal FAISS+BM25)

# Prediction
PREDICTION_METHOD = 'weighted_mean'  # or 'mean', 'median'
```

### 3. Run the System

```bash
python rag_price_prediction.py
```

**That's it!** No command-line arguments needed. The system will:
1. Load and preprocess data
2. Generate embeddings
3. Build FAISS and BM25 indices
4. Run all three retrieval methods
5. Generate predictions and LLM explanations
6. Create visualizations and save results

---

## 📊 Output Files

### Visualizations (4 files)
- **`comparison_all_modes.png`** - Side-by-side comparison of FAISS, BM25, and Hybrid
- **`detailed_faiss.png`** - 4-panel analysis (Actual vs Predicted, Error Distribution, Price Distribution, Metrics)
- **`detailed_bm25.png`** - 4-panel analysis for BM25
- **`detailed_hybrid.png`** - 4-panel analysis for Hybrid

### Data Files (3 files)
- **`predictions_with_similar_faiss.csv`** - Predictions with top 3 similar products
- **`predictions_with_similar_bm25.csv`**
- **`predictions_with_similar_hybrid.csv`**

**CSV Columns:**
- Product details (ID, title, brand, category)
- Prices (actual, predicted, error, error %)
- Similar products (titles, prices, similarity scores for top 3)
- Retrieval metadata (number of similar products, avg similarity)

### LLM Explanations (3 files)
- **`llm_explanations_faiss.txt`** - Detailed reasoning for 5 sample products
- **`llm_explanations_bm25.txt`**
- **`llm_explanations_hybrid.txt`**

### Summary Report
- **`complete_project_summary.txt`** - Full project report including:
  - Dataset statistics
  - Configuration details
  - Metrics comparison for all methods
  - Best performing method
  - List of all generated files

---

## 📈 Example Output

### Metrics Comparison
```
FAISS    | R²: 0.8542 | MAE: $45.23 | MAPE: 12.3%
BM25     | R²: 0.8234 | MAE: $52.67 | MAPE: 14.8%
HYBRID   | R²: 0.8891 | MAE: $38.91 | MAPE: 10.2%
```

### Sample LLM Explanation
```
================================================================================
PREDICTION EXPLANATION - HYBRID MODE
================================================================================

TARGET PRODUCT:
  Title: Dell XPS 13 9310 Touchscreen Laptop, 13.4 inch FHD...
  Brand: Dell
  Category: Computers
  
PREDICTION RESULTS:
  Actual Price:      $899.99
  Predicted Price:   $905.50
  Error:            $5.51 (0.6%)
  
RETRIEVAL ANALYSIS:
  Method:           HYBRID
  Similar Products: 15
  Avg Similarity:   0.847
  Price Range:      $850.00 - $950.00

TOP 5 MOST SIMILAR PRODUCTS:

1. Apple MacBook Pro 13-inch, M1 chip, 8GB RAM...
   Brand: Apple                | Price: $899.00  | Similarity: 0.945

2. Lenovo ThinkPad X1 Carbon Gen 9...
   Brand: Lenovo               | Price: $920.00  | Similarity: 0.912

3. HP Spectre x360 14 Convertible Laptop...
   Brand: HP                   | Price: $895.00  | Similarity: 0.887

4. ASUS ZenBook 13 Ultra-Slim Laptop...
   Brand: ASUS                 | Price: $880.00  | Similarity: 0.865

5. Microsoft Surface Laptop 4...
   Brand: Microsoft            | Price: $910.00  | Similarity: 0.843

--------------------------------------------------------------------------------
PREDICTION QUALITY ASSESSMENT:
  Quality Level: EXCELLENT ✓✓✓
  Reasoning: Prediction within 10% of actual price

SIMILARITY ANALYSIS:
  Very high similarity - retrieved products are highly comparable

PRICE REASONING:
  The predicted price of $905.50 was calculated using weighted_mean
  based on 15 similar products with average similarity 0.847.
  Prediction is $5.51 higher than actual - may be influenced by
  higher-priced similar items (range: $850.00-$950.00).

================================================================================
```

---

## 🔧 Customization

### Change Sample Size
For faster testing or more comprehensive analysis:
```python
SAMPLE_SIZE = 5000   # Quick test
SAMPLE_SIZE = 10000  # Balanced (recommended)
SAMPLE_SIZE = 50000  # Comprehensive
```

### Change Categories
```python
FILTER_CATEGORIES = ["Electronics"]           # Single category
FILTER_CATEGORIES = ["Computers", "Tablets"]  # Multiple categories
FILTER_CATEGORIES = None                      # No filter (all categories)
```

### Change Price Range
```python
MIN_PRICE = 10.0    # Minimum price
MAX_PRICE = 500.0   # Maximum price
```

### Tune Hybrid Retrieval
```python
ALPHA = 0.7   # More weight to FAISS
ALPHA = 0.3   # More weight to BM25
ALPHA = 0.5   # Equal weight (default)

K_BM25 = 30   # BM25 candidates
K_FAISS = 30  # FAISS candidates
K_HYBRID = 15 # Final number after fusion
```

### Change Prediction Method
```python
PREDICTION_METHOD = 'weighted_mean'  # Weight by similarity (default)
PREDICTION_METHOD = 'mean'           # Simple average
PREDICTION_METHOD = 'median'         # Median price
```

---

## 📖 How It Works

### 1. Data Loading
- Loads Amazon product metadata (JSONL format)
- Extracts: title, brand, features, description, price
- Filters by price range and category
- Splits into 80% train, 20% test

### 2. Text Processing
- Combines product fields into single text representation
- Format: `"Category: X ||| Brand: Y ||| Title: Z ||| Features: ... ||| Description: ..."`

### 3. Embedding Generation
- Uses SentenceTransformers (`all-MiniLM-L6-v2`)
- Generates 384-dimensional embeddings for each product
- Captures semantic meaning of product descriptions

### 4. Index Building

**FAISS Index:**
- Normalized L2 similarity search
- Fast approximate nearest neighbor retrieval

**BM25 Index:**
- Traditional keyword-based scoring
- Token-level matching with IDF weighting

### 5. Retrieval Methods

**FAISS:**
- Cosine similarity in embedding space
- Returns k most similar products

**BM25:**
- TF-IDF based scoring
- Best for exact keyword matches

**Hybrid:**
- Retrieves candidates from both FAISS and BM25
- Fuses scores: `hybrid_score = α × faiss_score + (1-α) × bm25_score`
- Optional reranking with embedding similarity

### 6. Price Prediction
- Retrieves k similar products
- Predicts price using weighted mean (by similarity score)
- Alternative methods: mean, median

### 7. LLM Reasoning
- Generates human-readable explanations
- Shows similar products that influenced prediction
- Assesses prediction quality
- Explains reasoning behind price estimate



## 👨‍💻 Author

**Masters Project - Information Retrieval and Storage**

Ujjwal Bhatta (ujjwalbhatta.89@gmail.com)

---

## 📄 License

This project is for academic purposes. Please cite appropriately if used in publications.

---

**Happy Price Predicting! 🎉**