import pandas as pd
import numpy as np
from complete_rag_pipeline import predict_product_performance
import re

# Load dataset
print("📂 Loading saved model/dataset...")
df = pd.read_csv("products_with_embeddings.csv")
print(f"✅ Loaded {len(df)} products")

print("="*80)
print("RAG SYSTEM EVALUATION")
print("="*80)

# Sample products to test
test_products = df.sample(n=10, random_state=42)

predictions = []
actuals = []

for idx, row in test_products.iterrows():
    print(f"\nTesting product {idx+1}/10...")

    # --- SAFE DESCRIPTION HANDLING ---
    desc = str(row.get('description', ""))  # ensures string, no crash on NaN

    # --- RAG Prediction Call ---
    result = predict_product_performance(
        category=row['category'],
        brand=row['brand'],
        description=desc[:100],  # first 100 chars for speed
        price=row['price']
    )

    prediction_text = result['prediction']

    # --- Extract predicted rating ---
    try:
        match = re.search(r'(\d+\.?\d*)/5', prediction_text)
        if match:
            predicted_rating = float(match.group(1))
        else:
            predicted_rating = result['avg_similar_rating']
    except:
        predicted_rating = result['avg_similar_rating']

    actual_rating = row['average_rating']

    predictions.append(predicted_rating)
    actuals.append(actual_rating)

    print(f"   Predicted: {predicted_rating:.2f}/5")
    print(f"   Actual:    {actual_rating:.2f}/5")
    print(f"   Error:     {abs(predicted_rating - actual_rating):.2f}")

# --- Evaluation Metrics ---
mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(predictions, actuals)]))

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Mean Absolute Error (MAE): {mae:.3f} stars")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} stars")

print("\nInterpretation:")
if mae < 0.5:
    print("✅ EXCELLENT: Predictions are very accurate!")
elif mae < 1.0:
    print("✅ GOOD: Predictions are reasonably accurate")
else:
    print("⚠️  NEEDS IMPROVEMENT: Predictions have significant error")

print("\nNote: Industry standard for rating prediction is MAE < 0.7")
print("="*80)
