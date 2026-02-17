import requests
from bs4 import BeautifulSoup
import time
import os

def scrape_paul_graham():
    # 1. Setup
    base_url = "http://www.paulgraham.com/"
    articles_url = base_url + "articles.html"
    output_file = "data/paul_graham.txt"
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"  Starting extraction from: {articles_url}")

    try:
        # Get the main list of articles
        response = requests.get(articles_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links ending in .html (PG's site is old school)
        links = [a['href'] for a in soup.find_all('a') if a['href'] and a['href'].endswith('.html')]
        
        # Filter out non-essays (index, rss, etc.)
        links = [l for l in links if 'index' not in l and 'rss' not in l]
        
        # Limit to first 50 essays for speed (remove [:50] to get everything)
        links = links[:50] 
        
        print(f" Found {len(links)} essays. Downloading...")
        
        all_text = ""
        
        for i, link in enumerate(links):
            full_url = base_url + link
            try:
                # Download specific essay
                r = requests.get(full_url)
                s = BeautifulSoup(r.text, 'html.parser')
                
                # Extract text (PG's site uses tables, this gets the raw text inside)
                text = s.get_text(separator=' ', strip=True)
                
                # Simple cleanup: Add a separator between essays
                all_text += text + "\n\n" + ("=" * 50) + "\n\n"
                
                print(f"   [{i+1}/{len(links)}] Downloaded: {link}")
                
                # Be polite to the server
                time.sleep(0.2)
                
            except Exception as e:
                print(f"    Failed to download {link}: {e}")

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
            
        print(f"\n✅ Success! Saved {len(all_text)} characters to {output_file}")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    scrape_paul_graham()