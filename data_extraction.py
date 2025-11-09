import json
import pandas as pd
from tqdm import tqdm

# ==== CONFIG ====
REVIEW_FILE = "/Users/ujjwalbhatta/Downloads/Electronics.jsonl"
META_FILE = "/Users/ujjwalbhatta/Downloads/meta_Electronics.jsonl"
OUTPUT_FILE = "products_for_rag.csv"
SAMPLE_SIZE = 5000

print("📦 Loading metadata...")
products = {}

with open(META_FILE, 'r') as f:
    for line in tqdm(f):
        item = json.loads(line.strip())
        parent_asin = item.get('parent_asin')
        
        if not parent_asin or not item.get('average_rating'):
            continue
        
        # Get brand
        details = item.get('details', {})
        brand = details.get('Brand') or item.get('store') or 'Unknown'
        
        # Get description
        desc = item.get('description', [])
        if isinstance(desc, list):
            desc = ' '.join(desc)
        
        products[parent_asin] = {
            'parent_asin': parent_asin,
            'category': item.get('main_category'),
            'title': item.get('title'),
            'description': str(desc)[:500],  # Limit length
            'price': item.get('price'),
            'brand': brand,
            'average_rating': item.get('average_rating'),  # TARGET
            'rating_count': item.get('rating_number'),
            'reviews': []  # Will store review texts
        }

print(f"✅ Loaded {len(products)} products")

print("\n📝 Adding reviews...")
with open(REVIEW_FILE, 'r') as f:
    for line in tqdm(f):
        review = json.loads(line.strip())
        parent_asin = review.get('parent_asin')
        
        if parent_asin in products and len(products[parent_asin]['reviews']) < 5:
            text = review.get('text', '')
            if text:
                products[parent_asin]['reviews'].append(text[:200])

print("✅ Reviews added")

print("\n🔧 Creating dataset...")
data = []
for asin, p in tqdm(products.items()):
    # Skip if missing required fields
    if not p['title'] or not p['price'] or not p['average_rating']:
        continue
    
    # Combine reviews
    review_text = ' '.join(p['reviews']) if p['reviews'] else ''
    
    data.append({
        'parent_asin': p['parent_asin'],
        'category': p['category'],
        'title': p['title'],
        'description': p['description'],
        'brand': p['brand'],
        'price': p['price'],
        'average_rating': p['average_rating'],
        'rating_count': p['rating_count'],
        'review_text': review_text
    })

df = pd.DataFrame(data)

# Convert price to numeric (some are strings)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Remove rows with bad data
df = df.dropna(subset=['price', 'average_rating', 'title'])

# Filter reasonable prices
df = df[(df['price'] > 0) & (df['price'] < 10000)]

# Sample if too large
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

print(f"\n✅ Final dataset: {len(df)} products")
print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
print(f"Rating range: {df['average_rating'].min():.1f} - {df['average_rating'].max():.1f}")

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Saved to {OUTPUT_FILE}")

# Show sample
print("\n📊 Sample:")
print(df[['title', 'price', 'average_rating', 'brand']].head())