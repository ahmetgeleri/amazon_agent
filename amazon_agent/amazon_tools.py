import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from langchain_core.tools import tool
import time
import random
import pandas as pd
import os
import re
from typing import Dict
import numpy as np

"""
Amazon Search Agent Toolkit
A set of tools for automated product discovery and data extraction from Amazon.com.tr
"""

@tool
def generate_amazon_search_url(search_query: str) -> str:
    """
    Generates a valid Amazon.com.tr search URL from a given search query.

    Args:
        search_query (str): The raw text or keywords to search for.

    Returns:
        str: A complete and URL-encoded Amazon search link.
    """
    base_url = "https://www.amazon.com.tr/s?k="
    query_encoded = quote_plus(search_query)
    return f"{base_url}{query_encoded}"

@tool
def fetch_search_results(url: str):
    """
    Sends an HTTP GET request to Amazon and retrieves the raw HTML content.
    If the request fails, tries to fetch again 3 times with delays.
    Args:
        url (str): The Amazon search results URL to fetch.

    Returns:
        str: The raw HTML source code of the page, or an error message if the request fails.
    """
    user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    for attempt in range(3):
        if attempt > 0:
            print(f"Retrying... (Attempt {attempt+1})")
            time.sleep(random.uniform(2, 5))  
        current_agent = random.choice(user_agents)
        headers = {
            "User-Agent": current_agent, 
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/", 
            "Connection": "keep-alive",
        }
        try:
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                with open("last_search.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                return "Success: HTML content saved to 'last_search.html'. You can now parse it."
        except Exception as e:
            print(f"Connection error: {str(e)}")
    return "Error: Failed to retrieve data after multiple attempts."

@tool
def parse_amazon_results(file_path: str = "last_search.html") -> str:
    """
    Parses the local HTML file using the exact logic provided,
    creates a DataFrame, calculates the sorting score,
    and saves the result to 'products_table.csv'.

    Args:
    html_content(str): The raw HTML string retrieved from an Amazon search page.
    
    Returns:
    str: A JSON-formatted string containing a list of titles, prices, and links for the top 10 products.
    """
    # Initialize parser and container 
    if not os.path.exists(file_path):
        return "Error: HTML file not found. Please fetch results first."
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    all_products = []
    items = soup.find_all("div", {"data-component-type": "s-search-result"})

    for item in items:
        # Extract Title 
        title_tag = item.find("h2")
        title = title_tag.text.strip() if title_tag else "N/A"

        # Extract Price 
        price_tag = item.find("span", {"class": "a-offscreen"})
        price = price_tag.text.strip() if price_tag else "N/A"

        # Extract Link 
        link_tag = item.find("a", {"class": "a-link-normal"})
        link = "N/A"
        if link_tag and 'href' in link_tag.attrs:
            raw_link = link_tag['href']
            link = f"https://www.amazon.com.tr{raw_link}" if raw_link.startswith('/') else raw_link
        
        # Extract Rating using regex 
        rating_tag = item.select_one("span.a-size-small.a-color-base")
        if rating_tag:
            raw_rating = rating_tag.text.strip() 
            match = re.search(r'(\d+(?:,\d+)?)', raw_rating)
            if match:
                rating = float(match.group(1).replace(',', '.'))
            else:
                rating = "N/A"
        else: 
            rating = "N/A"

        # Extract Comment Count using regex 
        review_tag = item.select_one("span.s-underline-text")
        if review_tag:
        
            rev_match = re.search(r'(\d+)', review_tag.text)
            review_count = int(rev_match.group(1)) if rev_match else 0
        else:
            review_count = 0

        # Extract Delivery Date
        delivery_tag = item.find("span", {"class": "a-text-bold"})
        delivery_date = delivery_tag.text.strip() if delivery_tag else "N/A"

        all_products.append({
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Number of Reviews": review_count,
            "Delivery Date": delivery_date,
            "Link": link
        })

    df = pd.DataFrame(all_products)
    if not df.empty:
        df.drop_duplicates(subset=['Title'], inplace=True)
        
        # Export data to CSV 
        df.to_csv("products_table.csv", index=False, encoding="utf-8-sig")
        
        print(f"Success! {len(df)} unique products saved.")
    else:
        print("Warning: No product data found.")
    return (
        f"ACTION REQUIRED: You MUST now call 'weighted_product_ranking'. "
        f"If the user did not specify preferences, use these default weights: "
        f'{{"Price": 0.4, "Rating": 0.4, "Number of Reviews": 0.2, "Delivery Date": 0.0}}'
    )
@tool
def weighted_product_ranking(weights: Dict[str, float], file_path: str = "products_table.csv"):
    """
    Ranks products based on dynamic weights.
    Parses Turkish delivery dates (e.g., "15 Oca") into numerical scores internally.
    Args:
        weights (Dict[str, float]): Weights for each criterion.
        file_path (str): Path to the CSV file containing product data.
    Returns:
        List of top 3 products as dictionaries.
    """
    # Normalize Weights
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # File Check
    if not os.path.exists(file_path):
        return "Error: Data source file not found."

    df = pd.read_csv(file_path)
    if df.empty:
        return "Error: DataFrame is empty."
    
    # Internal Price Cleaning Logic (Embedded)
    def clean_turkish_price(price_str):
        
        if pd.isna(price_str): return float('nan')
        s = str(price_str)

        s = s.replace("TL", "").replace("tl", "").strip()

        s = s.replace(".", "")

        s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return float('nan')
    
    # Internal Date Parsing Logic (Embedded)
    def parse_delivery_date(date_val) -> int:
        """Converts Turkish date string to an approximate Day of Year integer."""
        date_str = str(date_val).strip()
        
        PENALTY_SCORE = 366 

        if not date_str or date_str == "nan" or date_str == "N/A":
            return PENALTY_SCORE

        month_map = {
            "Oca": 1, "Şub": 2, "Mar": 3, "Nis": 4, "May": 5, "Haz": 6,
            "Tem": 7, "Ağu": 8, "Eyl": 9, "Eki": 10, "Kas": 11, "Ara": 12
        }

        try:
            match = re.search(r'(\d+).+?([a-zA-ZĞğÜüŞşİıÖö]+)', date_str)
            if match:
                day = int(match.group(1))
                month_abbr = match.group(2).capitalize()[:3]
                month_num = month_map.get(month_abbr, 12)
                return (month_num - 1) * 31 + day
        except Exception:
            pass
        
        return PENALTY_SCORE
    
    df['Price'] = df['Price'].apply(clean_turkish_price)
    df.dropna(subset=['Price'], inplace=True)

    df['delivery_score_val'] = df['Delivery Date'].apply(parse_delivery_date)
    
    numeric_cols = ["Price", "Rating", "Number of Reviews"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    epsilon = 1e-9 

    # Normalization 
    
    p_min, p_max = df['Price'].min(), df['Price'].max()
    df['n_price'] = (p_max - df['Price']) / (p_max - p_min + epsilon)

    d_min, d_max = df['delivery_score_val'].min(), df['delivery_score_val'].max()
    df['n_delivery'] = (d_max - df['delivery_score_val']) / (d_max - d_min + epsilon)    

    r_min, r_max = df['Rating'].min(), df['Rating'].max()
    df['n_rating'] = (df['Rating'] - r_min) / (r_max - r_min + epsilon)

    df['log_reviews'] = np.log1p(df['Number of Reviews']) 

    rv_min, rv_max = df['log_reviews'].min(), df['log_reviews'].max()
    df['n_reviews'] = (df['log_reviews'] - rv_min) / (rv_max - rv_min + epsilon)

    # Calculate Final Score
    df['Final_Score'] = (
        df['n_price'] * weights.get('Price', 0) +
        df['n_rating'] * weights.get('Rating', 0) +
        df['n_reviews'] * weights.get('Number of Reviews', 0) +
        df['n_delivery'] * weights.get('Delivery Date', 0)
    )

    # Return Top 3
    top_3 = df.sort_values(by='Final_Score', ascending=False).head(3)
    
    
    return top_3[["Title", "Price", "Rating","Number of Reviews", "Link", "Delivery Date", "Final_Score"]].to_dict(orient="records")

# Essential lists for the Agent to access tools
tools = [generate_amazon_search_url, fetch_search_results, parse_amazon_results, weighted_product_ranking]

