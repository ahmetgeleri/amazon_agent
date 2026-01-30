# Amazon Product Discovery Agent

This project is an AI-powered tool that automates product search and decision-making specifically for the amazon.com.tr marketplace. It uses Groq/Llama-3 to understand user queries, fetches real-time data from amazon.com.tr, and applies a mathematical weighted scoring algorithm to rank products based on price, ratings, and delivery date.

## Agent Capabilities & Key Features

- Agentic Orchestration (LCEL): Utilizes LangChain Expression Language to construct a deterministic execution. The agent manages state transitions between URL generation, data fetching, parsing, and analysis tools without human intervention.

- Semantic to Numeric Inference: Features a reasoning engine that translates qualitative user constraints (e.g., "budget friendly," "urgent delivery") into quantitative weight vectors. This allows the system to dynamically adjust its ranking algorithm based on semantic context.

- Hybrid Ranking Engine: Implements a composite scoring algorithm that merges LLM derived weights with statistical Min Max normalization. This ensures mathematically sound comparisons between disparate metrics such as Price, Rating, and Delivery Date.

- Context Aware Resource Management: Designed to bypass LLM token limits by offloading raw HTML processing to local storage. The agent selectively retrieves only structured metadata for the final inference stage, optimizing latency and cost.

- Marketplace Localization: The parsing logic includes specialized handlers for the amazon.com.tr infrastructure, autonomously managing regional currency normalization and date parsing formats.

## Tech Stack

- Python 3.14
- LangChain 
- Groq API 
- Numpy, Pandas & BeautifulSoup4 

## Installation

1. Install the required packages:
   pip install langchain langchain-groq pandas beautifulsoup4 requests

2. Export your Groq API Key

## Usage

1. Open 'amazon_agent.py' and modify the 'user_query' variable to set your search criteria.

2. Run the agent:
   python amazon_agent.py

The agent will execute the workflow on amazon.com.tr and display the top 3 recommended products in the terminal.

## Disclaimer

This project is for educational purposes only. Users are responsible for complying with Amazon's terms of service.