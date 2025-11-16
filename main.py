#!/usr/bin/env python3
"""
Complete RAG-Based Product Price Prediction System with LLM Reasoning
Supports: FAISS, BM25, Hybrid Retrieval + LLM Explanations
Masters Project - Information Retrieval and Storage

Features:
1. Three retrieval methods: FAISS, BM25, Hybrid
2. Comprehensive comparison across all methods
3. LLM-powered explanations for predictions
4. Detailed similar products analysis
5. Advanced visualizations and metrics
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# SIMPLE CONFIGURATION - Change these values!
# ============================================================================

FILE_PATH = '/Users/ujjwalbhatta/Downloads/meta_Electronics.jsonl'
OUTPUT_DIR = './complete_rag_outputs'

# Data settings
SAMPLE_SIZE = 12000
MIN_PRICE = 5.0
MAX_PRICE = 1000.0
FILTER_CATEGORIES = ["Computers"]

# Model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Retrieval settings
K_SIMILAR = 15      # For FAISS and BM25 modes
K_BM25 = 30         # For Hybrid mode
K_FAISS = 30        # For Hybrid mode
K_HYBRID = 15       # Final k for Hybrid mode
ALPHA = 0.5         # Hybrid weighting (alpha*FAISS + (1-alpha)*BM25)

# Prediction
PREDICTION_METHOD = 'weighted_mean'
RERANK_WITH_EMBEDDINGS = True

# LLM Explanations
NUM_EXPLANATION_SAMPLES = 5  # Number of products to explain in detail

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_amazon_data(file_path: str, sample_size: int, 
                     min_price: float, max_price: float,
                     filter_categories: List[str]) -> pd.DataFrame:
    """Load and filter Amazon product data"""
    products = []
    
    print(f"\n📁 Loading data from: {file_path}")
    print(f"   Filters: Price ${min_price}-${max_price}, Categories: {filter_categories}")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if len(products) >= sample_size:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Extract price
                if 'price' not in data or data['price'] is None:
                    continue
                
                try:
                    price = float(data['price'])
                except:
                    price = None
                    if isinstance(data.get('price', None), dict):
                        val = data['price'].get('value') or data['price'].get('amount')
                        if val:
                            price = float(val)
                    if price is None:
                        continue
                
                # Filter price range
                if price < min_price or price > max_price:
                    continue
                
                # Filter category
                category = data.get('main_category', data.get('category', 'Unknown'))
                if filter_categories and category not in filter_categories:
                    continue
                
                # Extract fields
                title = data.get('title') or data.get('product_title') or ''
                features = data.get('features') or data.get('feature_bullets') or []
                description = data.get('description') or data.get('product_description') or []
                
                # Extract brand
                brand = 'Unknown'
                details = data.get('details', {})
                if isinstance(details, dict):
                    brand = details.get('Brand', details.get('brand', 'Unknown'))
                
                product = {
                    'parent_asin': data.get('parent_asin', data.get('asin', '')),
                    'main_category': category,
                    'title': title,
                    'price': price,
                    'features': ' '.join(features) if isinstance(features, list) else str(features),
                    'description': ' '.join(description) if isinstance(description, list) else str(description),
                    'brand': brand or 'Unknown',
                    'store': data.get('store', 'Unknown')
                }
                
                products.append(product)
                
            except Exception:
                continue
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i+1} lines, collected {len(products)} products...")
    
    df = pd.DataFrame(products)
    df = df[df['title'].str.len() > 0].copy()
    df = df[(df['features'].str.len() > 0) | (df['description'].str.len() > 0)].copy()
    
    print(f"\n✅ Loaded {len(df)} products")
    print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"   Mean: ${df['price'].mean():.2f}, Median: ${df['price'].median():.2f}")
    
    return df


def create_combined_text(row: pd.Series) -> str:
    """Combine product information for embedding"""
    parts = [
        f"Category: {row['main_category']}",
        f"Brand: {row['brand']}",
        f"Title: {row['title']}",
        f"Features: {row['features']}",
        f"Description: {row['description']}"
    ]
    return ' ||| '.join(parts)


# ============================================================================
# STEP 2: EMBEDDINGS & INDICES
# ============================================================================

def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Generate embeddings using SentenceTransformers"""
    print(f"   Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"   Generating embeddings for {len(texts)} products...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.asarray(embeddings)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build FAISS index"""
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"   ✅ FAISS index: {index.ntotal} vectors (dim: {dimension})")
    return index


def build_bm25_index(train_df: pd.DataFrame) -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index"""
    corpus = train_df['combined_text'].fillna('').astype(str).tolist()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   ✅ BM25 index: {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus


# ============================================================================
# STEP 3: RETRIEVAL FUNCTIONS (FAISS, BM25, HYBRID)
# ============================================================================

def retrieve_similar_products_faiss(query_embedding: np.ndarray, 
                                   index: faiss.IndexFlatL2, 
                                   train_df: pd.DataFrame, 
                                   k: int) -> pd.DataFrame:
    """Retrieve using FAISS"""
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    similar = train_df.iloc[indices[0]].copy()
    similar['similarity_score'] = 1 - (distances[0] / 2)
    return similar


def retrieve_similar_products_bm25(query_text: str, 
                                   bm25: BM25Okapi,
                                   train_df: pd.DataFrame,
                                   k: int) -> pd.DataFrame:
    """Retrieve using BM25"""
    tokenized_query = query_text.split()
    scores = bm25.get_scores(tokenized_query)
    
    if np.all(np.isclose(scores, 0)):
        empty = train_df.iloc[:0].copy()
        empty['similarity_score'] = []
        return empty
    
    scores = np.array(scores, dtype=float)
    top_idx = np.argsort(scores)[::-1][:k]
    similar = train_df.iloc[top_idx].copy().reset_index(drop=True)
    
    max_score = scores[top_idx].max() if len(top_idx) > 0 else scores.max()
    if max_score <= 0:
        normalized = np.zeros_like(scores[top_idx])
    else:
        normalized = scores[top_idx] / (max_score + 1e-12)
    
    similar['similarity_score'] = normalized
    return similar


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0,1]"""
    if scores is None or len(scores) == 0:
        return np.array([])
    max_val = float(np.max(scores))
    if max_val <= 0:
        return np.zeros_like(scores)
    return scores / (max_val + 1e-12)


def retrieve_similar_products_hybrid(query_text: str,
                                    query_embedding: np.ndarray,
                                    bm25: BM25Okapi,
                                    faiss_index: faiss.IndexFlatL2,
                                    train_df: pd.DataFrame,
                                    train_embeddings: np.ndarray,
                                    k_bm25: int,
                                    k_faiss: int,
                                    k_final: int,
                                    alpha: float,
                                    rerank: bool) -> pd.DataFrame:
    """Hybrid retrieval combining BM25 and FAISS"""
    
    # BM25 retrieval
    tokenized_query = query_text.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = np.array(bm25_scores, dtype=float)
    
    if not np.all(bm25_scores <= 0):
        bm25_idx = np.argsort(bm25_scores)[::-1][:k_bm25]
        bm25_scores_norm = normalize_scores(bm25_scores[bm25_idx])
    else:
        bm25_idx = np.array([])
        bm25_scores_norm = np.array([])
    
    # FAISS retrieval
    q = query_embedding.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)
    distances, faiss_idx = faiss_index.search(q, k_faiss)
    
    faiss_idx_flat = faiss_idx[0]
    distances_flat = distances[0]
    faiss_sim = 1.0 - (distances_flat / 2.0)
    faiss_sim = np.clip(faiss_sim, 0.0, 1.0)
    
    # Merge candidates
    score_map = {}
    for idx, s in zip(faiss_idx_flat, faiss_sim):
        score_map[int(idx)] = {"faiss": float(s), "bm25": 0.0}
    
    for idx, s in zip(bm25_idx, bm25_scores_norm):
        idx = int(idx)
        if idx in score_map:
            score_map[idx]["bm25"] = float(s)
        else:
            score_map[idx] = {"faiss": 0.0, "bm25": float(s)}
    
    indices = np.array(list(score_map.keys()), dtype=int)
    faiss_s = np.array([score_map[i]["faiss"] for i in indices], dtype=float)
    bm25_s = np.array([score_map[i]["bm25"] for i in indices], dtype=float)
    
    hybrid = alpha * faiss_s + (1.0 - alpha) * bm25_s
    
    candidates = pd.DataFrame({"idx": indices, "hybrid_score": hybrid})
    candidates = candidates.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    
    # Optional reranking
    if rerank and train_embeddings is not None:
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        cand_embs = train_embeddings[candidates["idx"].values.astype(int)]
        cand_embs_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)
        cos_sims = (cand_embs_norm @ q_norm).astype(float)
        beta = 0.6
        new_scores = beta * normalize_scores(cos_sims) + (1.0 - beta) * normalize_scores(candidates["hybrid_score"].values)
        candidates["final_score"] = new_scores
        candidates = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
    else:
        candidates["final_score"] = candidates["hybrid_score"]
    
    top_candidates = candidates.head(k_final).copy()
    idxs = top_candidates["idx"].astype(int).values
    df_selected = train_df.iloc[idxs].copy().reset_index(drop=True)
    df_selected["similarity_score"] = top_candidates["final_score"].values
    
    return df_selected


# ============================================================================
# STEP 4: PRICE PREDICTION
# ============================================================================

def predict_price(similar_products: pd.DataFrame, method: str) -> Tuple[float, Dict]:
    """Predict price from similar products"""
    if similar_products is None or len(similar_products) == 0:
        return float('nan'), {}
    
    prices = similar_products['price'].values.astype(float)
    similarities = similar_products['similarity_score'].values.astype(float)
    
    if method == 'weighted_mean':
        if similarities.sum() == 0:
            predicted_price = prices.mean()
        else:
            weights = similarities / similarities.sum()
            predicted_price = float(np.sum(prices * weights))
    elif method == 'median':
        predicted_price = float(np.median(prices))
    else:  # mean
        predicted_price = float(np.mean(prices))
    
    metadata = {
        'method': method,
        'similar_count': len(similar_products),
        'price_range': (float(prices.min()), float(prices.max())),
        'avg_similarity': float(similarities.mean()),
        'top_similarity': float(similarities.max())
    }
    
    return predicted_price, metadata


# ============================================================================
# STEP 5: LLM REASONING
# ============================================================================

def generate_llm_explanation(test_product: Dict, 
                             similar_products: pd.DataFrame,
                             predicted_price: float,
                             true_price: float,
                             mode: str) -> str:
    """Generate detailed explanation for price prediction"""
    
    avg_sim = similar_products['similarity_score'].mean()
    price_range = (similar_products['price'].min(), similar_products['price'].max())
    error = abs(predicted_price - true_price)
    error_pct = (error / true_price) * 100
    
    explanation = f"""
{'='*80}
PREDICTION EXPLANATION - {mode.upper()} MODE
{'='*80}

TARGET PRODUCT:
  Title: {test_product['title'][:70]}...
  Brand: {test_product['brand']}
  Category: {test_product['main_category']}
  
PREDICTION RESULTS:
  Actual Price:      ${true_price:.2f}
  Predicted Price:   ${predicted_price:.2f}
  Error:            ${error:.2f} ({error_pct:.1f}%)
  
RETRIEVAL ANALYSIS:
  Method:           {mode.upper()}
  Similar Products: {len(similar_products)}
  Avg Similarity:   {avg_sim:.3f}
  Price Range:      ${price_range[0]:.2f} - ${price_range[1]:.2f}

TOP 5 MOST SIMILAR PRODUCTS:
"""
    
    for i, (idx, row) in enumerate(similar_products.head(5).iterrows(), 1):
        explanation += f"\n{i}. {row['title'][:65]}..."
        explanation += f"\n   Brand: {row['brand']:20s} | Price: ${row['price']:7.2f} | Similarity: {row['similarity_score']:.3f}"
    
    # Add prediction quality assessment
    explanation += "\n\n" + "-"*80 + "\n"
    explanation += "PREDICTION QUALITY ASSESSMENT:\n"
    
    if error_pct < 10:
        quality = "EXCELLENT ✓✓✓"
        reason = "Prediction within 10% of actual price"
    elif error_pct < 20:
        quality = "GOOD ✓✓"
        reason = "Prediction within 20% of actual price"
    elif error_pct < 30:
        quality = "FAIR ✓"
        reason = "Prediction within 30% of actual price"
    else:
        quality = "POOR ✗"
        reason = "Prediction differs significantly from actual"
    
    explanation += f"  Quality Level: {quality}\n"
    explanation += f"  Reasoning: {reason}\n\n"
    
    # Add similarity analysis
    if avg_sim > 0.8:
        sim_analysis = "Very high similarity - retrieved products are highly comparable"
    elif avg_sim > 0.6:
        sim_analysis = "Good similarity - retrieved products are reasonably comparable"
    elif avg_sim > 0.4:
        sim_analysis = "Moderate similarity - some comparable products found"
    else:
        sim_analysis = "Low similarity - limited comparable products in dataset"
    
    explanation += f"SIMILARITY ANALYSIS:\n  {sim_analysis}\n"
    
    # Add price reasoning
    explanation += f"\nPRICE REASONING:\n"
    explanation += f"  The predicted price of ${predicted_price:.2f} was calculated using {PREDICTION_METHOD}\n"
    explanation += f"  based on {len(similar_products)} similar products with average similarity {avg_sim:.3f}.\n"
    
    if predicted_price > true_price:
        explanation += f"  Prediction is ${error:.2f} higher than actual - may be influenced by\n"
        explanation += f"  higher-priced similar items (range: ${price_range[0]:.2f}-${price_range[1]:.2f}).\n"
    elif predicted_price < true_price:
        explanation += f"  Prediction is ${error:.2f} lower than actual - may be influenced by\n"
        explanation += f"  lower-priced similar items (range: ${price_range[0]:.2f}-${price_range[1]:.2f}).\n"
    else:
        explanation += f"  Prediction matches actual price very closely!\n"
    
    explanation += "\n" + "="*80 + "\n"
    
    return explanation


# ============================================================================
# STEP 6: EVALUATION
# ============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'R²': float('nan'),
                'MAPE': float('nan'), 'Median_AE': float('nan')}
    
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    
    mae = mean_absolute_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
    r2 = r2_score(y_true_f, y_pred_f)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true_f - y_pred_f) / y_true_f)) * 100
        if not np.isfinite(mape):
            mape = float('nan')
    
    median_ae = np.median(np.abs(y_true_f - y_pred_f))
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R²': float(r2),
        'MAPE': float(mape),
        'Median_AE': float(median_ae)
    }


# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

def plot_comparison(results_dict: Dict[str, pd.DataFrame], output_dir: str):
    """Create comparison plot for all three modes"""
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    colors = {'faiss': 'steelblue', 'bm25': 'forestgreen', 'hybrid': 'darkorange'}
    
    for idx, (mode, df) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(df['true_price'], df['predicted_price'], 
                  alpha=0.6, s=50, color=colors[mode], 
                  edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_p = min(df['true_price'].min(), df['predicted_price'].min())
        max_p = max(df['true_price'].max(), df['predicted_price'].max())
        ax.plot([min_p, max_p], [min_p, max_p], 
                'r--', lw=2.5, label='Perfect', alpha=0.8)
        
        # Calculate metrics
        mask = ~np.isnan(df['predicted_price'].values)
        if mask.sum() > 0:
            r2 = r2_score(df['true_price'].values[mask], df['predicted_price'].values[mask])
            mae = mean_absolute_error(df['true_price'].values[mask], df['predicted_price'].values[mask])
            rmse = np.sqrt(mean_squared_error(df['true_price'].values[mask], df['predicted_price'].values[mask]))
            
            textstr = f'R² = {r2:.4f}\nMAE = ${mae:.2f}\nRMSE = ${rmse:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace')
        
        ax.set_xlabel('Actual Price ($)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Price ($)', fontsize=13, fontweight='bold')
        ax.set_title(f'{mode.upper()} Mode', fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('Price Prediction Comparison: FAISS vs BM25 vs Hybrid', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/comparison_all_modes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    plt.close()


def plot_individual_mode(results_df: pd.DataFrame, output_dir: str, mode: str):
    """Create detailed plot for individual mode"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(results_df['true_price'], results_df['predicted_price'], 
               alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
    min_p = min(results_df['true_price'].min(), results_df['predicted_price'].min())
    max_p = max(results_df['true_price'].max(), results_df['predicted_price'].max())
    ax1.plot([min_p, max_p], [min_p, max_p], 'r--', lw=2, label='Perfect')
    
    mask = ~np.isnan(results_df['predicted_price'].values)
    if mask.sum() > 0:
        r2 = r2_score(results_df['true_price'].values[mask], 
                     results_df['predicted_price'].values[mask])
        mae = mean_absolute_error(results_df['true_price'].values[mask], 
                                  results_df['predicted_price'].values[mask])
        textstr = f'R² = {r2:.4f}\nMAE = ${mae:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
    ax1.set_xlabel('Actual Price ($)', fontweight='bold')
    ax1.set_ylabel('Predicted Price ($)', fontweight='bold')
    ax1.set_title(f'Actual vs Predicted ({mode.upper()})', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Error Distribution
    ax2 = axes[0, 1]
    errors = results_df['predicted_price'] - results_df['true_price']
    ax2.hist(errors, bins=50, color='coral', edgecolor='darkred', alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error ($)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Error Distribution', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Price Distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['true_price'], bins=40, alpha=0.6, label='Actual', color='green')
    ax3.hist(results_df['predicted_price'], bins=40, alpha=0.6, label='Predicted', color='orange')
    ax3.set_xlabel('Price ($)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Price Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics = evaluate_predictions(results_df['true_price'].values, 
                                   results_df['predicted_price'].values)
    
    metrics_text = f"METRICS - {mode.upper()}\n" + "="*35 + "\n\n"
    metrics_text += f"MAE:  ${metrics['MAE']:.2f}\n"
    metrics_text += f"RMSE: ${metrics['RMSE']:.2f}\n"
    metrics_text += f"R²:   {metrics['R²']:.4f}\n"
    metrics_text += f"MAPE: {metrics['MAPE']:.2f}%\n"
    metrics_text += f"Median AE: ${metrics['Median_AE']:.2f}\n"
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_path = f'{output_dir}/detailed_{mode}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: detailed_{mode}.png")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_single_mode(mode: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   train_embeddings: np.ndarray, test_embeddings: np.ndarray,
                   faiss_index, bm25_index) -> Tuple[pd.DataFrame, List]:
    """Run prediction for a single mode"""
    
    print(f"\n{'='*80}")
    print(f"Running {mode.upper()} Mode")
    print(f"{'='*80}")
    
    predictions = []
    all_similar = []
    retrieval_times = []
    
    start_time = time.time()
    
    for i, row in test_df.iterrows():
        t0 = time.time()
        
        if mode == 'faiss':
            similar = retrieve_similar_products_faiss(
                test_embeddings[i], faiss_index, train_df, K_SIMILAR
            )
        elif mode == 'bm25':
            similar = retrieve_similar_products_bm25(
                row['combined_text'], bm25_index, train_df, K_SIMILAR
            )
        else:  # hybrid
            similar = retrieve_similar_products_hybrid(
                row['combined_text'], test_embeddings[i],
                bm25_index, faiss_index, train_df, train_embeddings,
                K_BM25, K_FAISS, K_HYBRID, ALPHA, RERANK_WITH_EMBEDDINGS
            )
        
        t1 = time.time()
        retrieval_times.append(t1 - t0)
        
        pred_price, _ = predict_price(similar, PREDICTION_METHOD)
        predictions.append(pred_price)
        all_similar.append(similar)
        
        if (i + 1) % 100 == 0:
            avg_time = np.mean(retrieval_times[-100:])
            print(f"   Processed {i+1}/{len(test_df)} | Avg time: {avg_time:.4f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed {mode.upper()} mode in {total_time:.2f}s")
    print(f"   Average retrieval time: {np.mean(retrieval_times):.4f}s per query")
    
    results_df = test_df.copy()
    results_df['predicted_price'] = predictions
    results_df['true_price'] = results_df['price']
    
    # Evaluate
    metrics = evaluate_predictions(results_df['true_price'].values, 
                                   results_df['predicted_price'].values)
    
    print(f"\n   Metrics for {mode.upper()}:")
    print(f"   MAE:  ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   R²:   {metrics['R²']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    return results_df, all_similar


def main():
    """Main execution pipeline - runs all three modes"""
    
    print("="*80)
    print("COMPLETE RAG PRICE PREDICTION WITH LLM REASONING")
    print("FAISS + BM25 + HYBRID + LLM Explanations")
    print("Masters Project - Information Retrieval and Storage")
    print("="*80)
    
    # STEP 1: Load data
    print("\n[STEP 1] Loading Amazon product data...")
    df = load_amazon_data(FILE_PATH, SAMPLE_SIZE, MIN_PRICE, MAX_PRICE, FILTER_CATEGORIES)
    
    if len(df) < 100:
        print("❌ Not enough data. Check file path and filters.")
        return
    
    # Create combined text
    print("\n[STEP 2] Creating text representations...")
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    
    # Split data
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy().reset_index(drop=True)
    test_df = df.iloc[train_size:].copy().reset_index(drop=True)
    
    print(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    
    # STEP 2: Generate embeddings
    print("\n[STEP 3] Generating embeddings...")
    train_embeddings = generate_embeddings(train_df['combined_text'].tolist(), EMBEDDING_MODEL)
    test_embeddings = generate_embeddings(test_df['combined_text'].tolist(), EMBEDDING_MODEL)
    print(f"   ✅ Train embeddings: {train_embeddings.shape}")
    print(f"   ✅ Test embeddings: {test_embeddings.shape}")
    
    # STEP 3: Build indices
    print("\n[STEP 4] Building indices...")
    print("   Building FAISS index...")
    faiss_index = build_faiss_index(train_embeddings)
    print("   Building BM25 index...")
    bm25_index, _ = build_bm25_index(train_df)
    
    # STEP 4: Run all three modes
    print("\n[STEP 5] Running all retrieval modes...")
    
    results_dict = {}
    all_similar_dict = {}
    
    for mode in ['faiss', 'bm25', 'hybrid']:
        results_df, all_similar = run_single_mode(
            mode, train_df, test_df, 
            train_embeddings, test_embeddings,
            faiss_index, bm25_index
        )
        results_dict[mode] = results_df
        all_similar_dict[mode] = all_similar
    
    # STEP 5: Generate LLM explanations
    print("\n[STEP 6] Generating LLM explanations for sample products...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Select random samples
    sample_indices = np.random.choice(len(test_df), 
                                     min(NUM_EXPLANATION_SAMPLES, len(test_df)), 
                                     replace=False)
    
    # Generate explanations for each mode
    for mode in ['faiss', 'bm25', 'hybrid']:
        explanation_file = os.path.join(OUTPUT_DIR, f'llm_explanations_{mode}.txt')
        
        with open(explanation_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"LLM EXPLANATIONS - {mode.upper()} MODE\n")
            f.write(f"{'='*80}\n\n")
            
            for idx in sample_indices:
                test_product = test_df.iloc[idx].to_dict()
                similar = all_similar_dict[mode][idx]
                pred_price = results_dict[mode].iloc[idx]['predicted_price']
                true_price = results_dict[mode].iloc[idx]['true_price']
                
                explanation = generate_llm_explanation(
                    test_product, similar, pred_price, true_price, mode
                )
                
                f.write(explanation + "\n\n")
        
        print(f"   ✅ Saved: llm_explanations_{mode}.txt")
    
    # STEP 6: Create visualizations
    print("\n[STEP 7] Creating visualizations...")
    
    # Comparison plot
    plot_comparison(results_dict, OUTPUT_DIR)
    
    # Individual detailed plots
    for mode in ['faiss', 'bm25', 'hybrid']:
        plot_individual_mode(results_dict[mode], OUTPUT_DIR, mode)
    
    # STEP 7: Save comprehensive results
    print("\n[STEP 8] Saving results...")
    
    for mode in ['faiss', 'bm25', 'hybrid']:
        # Save predictions with similar products
        results = []
        for idx, row in results_dict[mode].iterrows():
            similar = all_similar_dict[mode][idx]
            result = {
                'product_id': row['parent_asin'],
                'title': row['title'][:100],
                'brand': row['brand'],
                'category': row['main_category'],
                'true_price': row['true_price'],
                'predicted_price': row['predicted_price'],
                'error': row['predicted_price'] - row['true_price'],
                'abs_error': abs(row['predicted_price'] - row['true_price']),
                'error_pct': abs((row['predicted_price'] - row['true_price']) / row['true_price'] * 100),
                'num_similar': len(similar),
                'avg_similarity': similar['similarity_score'].mean() if len(similar) > 0 else 0,
                'similar_1_title': similar.iloc[0]['title'][:50] if len(similar) > 0 else '',
                'similar_1_price': similar.iloc[0]['price'] if len(similar) > 0 else 0,
                'similar_1_sim': similar.iloc[0]['similarity_score'] if len(similar) > 0 else 0,
                'similar_2_title': similar.iloc[1]['title'][:50] if len(similar) > 1 else '',
                'similar_2_price': similar.iloc[1]['price'] if len(similar) > 1 else 0,
                'similar_2_sim': similar.iloc[1]['similarity_score'] if len(similar) > 1 else 0,
                'similar_3_title': similar.iloc[2]['title'][:50] if len(similar) > 2 else '',
                'similar_3_price': similar.iloc[2]['price'] if len(similar) > 2 else 0,
                'similar_3_sim': similar.iloc[2]['similarity_score'] if len(similar) > 2 else 0,
            }
            results.append(result)
        
        results_df_full = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, f'predictions_with_similar_{mode}.csv')
        results_df_full.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: predictions_with_similar_{mode}.csv")
    
    # STEP 8: Create comprehensive summary
    print("\n[STEP 9] Creating comprehensive summary report...")
    
    summary_path = os.path.join(OUTPUT_DIR, 'complete_project_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE RAG PRICE PREDICTION PROJECT - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROJECT OVERVIEW:\n")
        f.write(f"  Dataset: {FILE_PATH}\n")
        f.write(f"  Total products loaded: {len(df)}\n")
        f.write(f"  Train set: {len(train_df)}\n")
        f.write(f"  Test set: {len(test_df)}\n")
        f.write(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}\n")
        f.write(f"  Category: {FILTER_CATEGORIES}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Embedding model: {EMBEDDING_MODEL}\n")
        f.write(f"  Prediction method: {PREDICTION_METHOD}\n")
        f.write(f"  k (FAISS/BM25): {K_SIMILAR}\n")
        f.write(f"  k_hybrid: {K_HYBRID}\n")
        f.write(f"  Alpha (hybrid weight): {ALPHA}\n")
        f.write(f"  Reranking: {RERANK_WITH_EMBEDDINGS}\n\n")
        
        f.write("="*80 + "\n")
        f.write("EVALUATION METRICS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        for mode in ['faiss', 'bm25', 'hybrid']:
            metrics = evaluate_predictions(
                results_dict[mode]['true_price'].values,
                results_dict[mode]['predicted_price'].values
            )
            
            f.write(f"{mode.upper()} MODE:\n")
            f.write(f"  MAE:        ${metrics['MAE']:.2f}\n")
            f.write(f"  RMSE:       ${metrics['RMSE']:.2f}\n")
            f.write(f"  R²:         {metrics['R²']:.4f}\n")
            f.write(f"  MAPE:       {metrics['MAPE']:.2f}%\n")
            f.write(f"  Median AE:  ${metrics['Median_AE']:.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST PERFORMING MODE:\n")
        f.write("="*80 + "\n\n")
        
        # Find best mode by R²
        best_mode = max(['faiss', 'bm25', 'hybrid'], 
                       key=lambda m: evaluate_predictions(
                           results_dict[m]['true_price'].values,
                           results_dict[m]['predicted_price'].values
                       )['R²'])
        
        best_metrics = evaluate_predictions(
            results_dict[best_mode]['true_price'].values,
            results_dict[best_mode]['predicted_price'].values
        )
        
        f.write(f"Best Mode: {best_mode.upper()}\n")
        f.write(f"R² Score: {best_metrics['R²']:.4f}\n")
        f.write(f"MAE: ${best_metrics['MAE']:.2f}\n")
        f.write(f"MAPE: {best_metrics['MAPE']:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("GENERATED FILES:\n")
        f.write("="*80 + "\n")
        f.write("  1. comparison_all_modes.png - Side-by-side comparison\n")
        f.write("  2. detailed_faiss.png - FAISS detailed analysis\n")
        f.write("  3. detailed_bm25.png - BM25 detailed analysis\n")
        f.write("  4. detailed_hybrid.png - Hybrid detailed analysis\n")
        f.write("  5. predictions_with_similar_faiss.csv - FAISS results\n")
        f.write("  6. predictions_with_similar_bm25.csv - BM25 results\n")
        f.write("  7. predictions_with_similar_hybrid.csv - Hybrid results\n")
        f.write("  8. llm_explanations_faiss.txt - FAISS explanations\n")
        f.write("  9. llm_explanations_bm25.txt - BM25 explanations\n")
        f.write("  10. llm_explanations_hybrid.txt - Hybrid explanations\n")
        f.write("  11. complete_project_summary.txt - This file\n")
    
    print(f"   ✅ Saved: complete_project_summary.txt")
    
    # Final summary
    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print(f"\nAll results saved in: {OUTPUT_DIR}/")
    print("\nGenerated Files:")
    print("  📊 Visualizations:")
    print("     - comparison_all_modes.png (FAISS vs BM25 vs Hybrid)")
    print("     - detailed_faiss.png, detailed_bm25.png, detailed_hybrid.png")
    print("  📄 Data Files:")
    print("     - predictions_with_similar_[mode].csv (includes top 3 similar products)")
    print("  📝 LLM Explanations:")
    print("     - llm_explanations_[mode].txt (detailed reasoning)")
    print("  📋 Summary:")
    print("     - complete_project_summary.txt")
    
    print("\n" + "="*80)
    print("METRICS COMPARISON SUMMARY:")
    print("="*80)
    
    for mode in ['faiss', 'bm25', 'hybrid']:
        metrics = evaluate_predictions(
            results_dict[mode]['true_price'].values,
            results_dict[mode]['predicted_price'].values
        )
        print(f"\n{mode.upper():8s} | R²: {metrics['R²']:.4f} | MAE: ${metrics['MAE']:.2f} | MAPE: {metrics['MAPE']:.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()