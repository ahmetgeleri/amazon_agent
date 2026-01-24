import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from langchain_core.tools import tool
import json

@tool
def generate_amazon_search_url(search_query: str) -> str:
    """
    Generates a valid Amazon.com.tr search URL from a given search query.
    """
    base_url = "https://www.amazon.com.tr/s?k="
    query_encoded = quote_plus(search_query)
    return f"{base_url}{query_encoded}"

@tool
def fetch_search_results(url: str):
    """
    Sends a request to the given Amazon URL and returns the raw HTML content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error: Amazon responded with status code {response.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

@tool
def parse_amazon_results(html_content: str) -> str:
    """
    Parses Amazon HTML content and extracts product titles, prices, and links in JSON format.
    """
    if not html_content or "Error" in html_content:
        return "No valid content to parse."

    soup = BeautifulSoup(html_content, 'html.parser')
    products = []
    # Scraping logic targeting Amazon's search result containers
    items = soup.find_all("div", {"data-component-type": "s-search-result"})

    for item in items[:10]: # Limit to top 10 results
        title_tag = item.find("h2")
        title = title_tag.text.strip() if title_tag else "No title found"

        # Targeting the 'a-offscreen' class for clean price data as discussed
        price_tag = item.find("span", {"class": "a-offscreen"})
        price = price_tag.text.strip() if price_tag else "No price info"

        link_tag = item.find("a", {"class": "a-link-normal"})
        if link_tag and 'href' in link_tag.attrs:
            raw_link = link_tag['href']
            link = f"https://www.amazon.com.tr{raw_link}" if raw_link.startswith('/') else raw_link
        else:
            link = "No link found"

        products.append({"title": title, "price": price, "link": link})

    return json.dumps(products, ensure_ascii=False)

# Essential lists for the Agent to access tools
tools = [generate_amazon_search_url, fetch_search_results, parse_amazon_results]
# tool_mapping is used by execute_tool to find the function object by its string name
tool_mapping = {tool.name: tool for tool in tools}