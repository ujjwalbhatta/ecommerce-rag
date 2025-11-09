import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# ==== LOAD SAVED ARTIFACTS ====
print("📂 Loading saved model...")

df = pd.read_csv("products_with_embeddings.csv")
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.bin")

with open("model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

model = SentenceTransformer(model_info['model_name'])

print(f"✅ Loaded {len(df)} products")

# ==== RETRIEVAL FUNCTION ====
def retrieve_similar_products(query_text, top_k=5):
    """Find similar products from database"""
    query_embedding = model.encode([query_text])
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]
    return results

# ==== RAG WITH LLM ====
def predict_product_performance(category, brand, description, price):
    """
    RAG Pipeline: Retrieve similar products + Generate prediction
    """
    
    # 1. Create query for new product
    query = f"Category: {category} | Brand: {brand} | Product: {description} | Price: ${price}"
    
    print(f"🔍 Searching for similar products...")
    similar_products = retrieve_similar_products(query, top_k=5)
    
    # 2. Analyze retrieved products
    avg_rating = similar_products['average_rating'].mean()
    avg_price = similar_products['price'].mean()
    ratings_list = similar_products['average_rating'].tolist()
    
    # 3. Prepare context for LLM
    context = "Similar products found:\n\n"
    for i, (idx, row) in enumerate(similar_products.iterrows(), 1):
        context += f"{i}. {row['title'][:80]}\n"
        context += f"   Price: ${row['price']:.2f} | Rating: {row['average_rating']}/5 ({row['rating_count']} reviews)\n"
        context += f"   Brand: {row['brand']}\n\n"
    
    # Extract key specs from description for comparison
    battery_mention = "battery" in description.lower()
    
    # 4. Create improved prompt for LLM
    prompt = f"""You are a critical product analyst. Predict realistic performance for a new product by comparing to similar products. Be HONEST about problems.

NEW PRODUCT:
- Category: {category}
- Brand: {brand} {"(UNKNOWN BRAND - HIGH RISK)" if brand not in ['Sony', 'Apple', 'Samsung', 'Bose', 'JBL', 'Anker'] else "(ESTABLISHED BRAND)"}
- Description: {description}
- Price: ${price:.2f}

SIMILAR PRODUCTS FOR COMPARISON:
{context}

CRITICAL ANALYSIS:
- Similar products average rating: {avg_rating:.2f}/5
- Similar products average price: ${avg_price:.2f}
- Your product is {"MUCH MORE EXPENSIVE" if price > avg_price * 1.5 else "CHEAPER" if price < avg_price * 0.7 else "similarly priced"}
- Price difference: ${price - avg_price:+.2f} ({((price/avg_price - 1) * 100):+.1f}%)

IMPORTANT: Consider these risk factors:
1. Is the brand trusted? Unknown brands at high prices get poor reviews
2. Are the specs competitive? (battery life, features vs. price)
3. Is the price reasonable for what's offered?
4. What do similar products' reviews complain about?

Provide a REALISTIC prediction:
Expected Rating: [Be honest - unknown brands or bad specs = low ratings]
Confidence: [Low/Medium/High]
Risk: [Consider price, brand recognition, and specs]
Reasoning: [2-3 sentences explaining WHY, mention specific concerns]

Be critical and realistic, not optimistic!
"""
    
    # 5. Call LLM with Ollama (Local, Free)
    print("\n🤖 Calling LLM (this may take 10-30 seconds)...")
    
    try:
        import ollama
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        prediction = response['message']['content']
        
        print("\n" + "="*80)
        print("🎯 LLM PREDICTION:")
        print("="*80)
        print(prediction)
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error calling Ollama: {e}")
        print("\nMake sure:")
        print("1. Ollama is running: ollama serve")
        print("2. Model is downloaded: ollama pull llama3.2")
        print("3. Python library installed: pip install ollama")
        prediction = None
    
    return {
        'query': query,
        'similar_products': similar_products,
        'prompt': prompt,
        'prediction': prediction,
        'avg_similar_rating': avg_rating
    }

# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    print("\n🎯 RAG PREDICTION EXAMPLE\n")
    
    # Test with a new product
    result = predict_product_performance(
        category="Electronics",
        brand="Bhatta",
        description="Wireless noise-canceling headphones with 3-hour battery life",
        price=799.99
    )
    
    print(f"\n📊 Quick Stats:")
    print(f"Average rating of similar products: {result['avg_similar_rating']:.2f}/5")
    print(f"\nFound {len(result['similar_products'])} similar products")
