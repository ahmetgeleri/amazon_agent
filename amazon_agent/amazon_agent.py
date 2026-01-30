from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from amazon_tools import *
from langchain_groq import ChatGroq

tool_mapping = {tool.name: tool for tool in tools}

# LLM Setup
llm = ChatGroq(
    groq_api_key="API_KEY", 
    model_name="llama-3.1-8b-instant", 
    temperature=0.1
)

# Binding the tools to the LLM to enable tool usage capabilities
llm_with_tools = llm.bind_tools(tools)

system_prompt = """
You are an autonomous Amazon shopping assistant. 
You must execute the following workflow step-by-step for the user's query:
1. Generate the Amazon search URL.
2. Fetch the search results (this saves data to a file).
3. Parse the saved HTML results.
4. Rank the products using weighted scoring based on user preferences.
5. You must use these exact keys: 'Price', 'Rating', 'Number of Reviews', 'Delivery Date'.
Do not ask questions. Just execute the tools in order.
"""

def execute_tool(tool_call):
    """
    Executes tool calls.
    For ranking, displays a clean customer-facing list (Title, Price, Delivery, Link).
    """
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    call_id = tool_call["id"]
    
    # Minimal Log
    print(f"\nTool: {tool_name}")
    try:
        print(f"Args: {json.dumps(tool_args, ensure_ascii=False)}")
    except:
        print(f"Args: {tool_args}")

    # Validation
    if tool_name not in tool_mapping:
        err_msg = f"Error: Tool '{tool_name}' not found."
        print(err_msg)
        return ToolMessage(content=err_msg, tool_call_id=call_id)
    
    # Execution
    try:
        target_tool = tool_mapping[tool_name]
        output = target_tool.invoke(tool_args)
        
        # Customer Facing Display
        if tool_name == "weighted_product_ranking" and isinstance(output, list):
            print("\nTop 3 Recommendations:")
            for i, item in enumerate(output, 1):
                title = item.get('Title', 'N/A')
                price = item.get('Price', 'N/A')
                delivery = item.get('Delivery Date', 'N/A')
                link = item.get('Link', 'N/A')
                rating = item.get('Rating', 'N/A')
                reviews = item.get('Number of Reviews', 'N/A')
                
                # Clean Block Format
                print(f"{i}. {title}")
                print(f"   Price:    {price} TL")
                print(f"   Delivery: {delivery}")
                print(f"   Rating:   {rating} stars ({reviews} reviews)")
                print(f"   Link:     {link}")
                print("") 
        
        # Standard Display for other tools
        else:
            out_str = str(output)
            print(f"Output: {out_str[:150]}..." if len(out_str) > 150 else f"Output: {out_str}")
            
    except Exception as e:
        output = f"Execution Error: {str(e)}"
        print(output)

    return ToolMessage(content=str(output), tool_call_id=call_id)

# LCEL Chain Construction
summarization_chain = (
    # Starting initial query
    RunnablePassthrough.assign(
        messages = lambda x: [SystemMessage(content=system_prompt), HumanMessage(content = x["query"])]
    )
    # First LLM call (generate_amazon_search_url)
    | RunnablePassthrough.assign(
        ai_response = lambda x:llm_with_tools.invoke(x["messages"])
    )
    # Process first tool call
    | RunnablePassthrough.assign(
        tool_messages = lambda x: [
            execute_tool(tc) for tc in x["ai_response"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages = lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    # Second LLM call (fetch_search_results)
    | RunnablePassthrough.assign(
        ai_response2 = lambda x: llm_with_tools.invoke(x["messages"])
    )    
    # Process second tool call
    | RunnablePassthrough.assign(
        tool_messages2 = lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages = lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    # Third LLM call (parse_amazon_results)
    | RunnablePassthrough.assign(
        ai_response3 = lambda x: llm_with_tools.invoke(x["messages"])
    )    
    # Process third tool call
    | RunnablePassthrough.assign(
        tool_messages3 = lambda x: [
            execute_tool(tc) for tc in x["ai_response3"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages = lambda x: x["messages"] + [x["ai_response3"]] + x["tool_messages3"]
    )
    # Fourth LLM call (weighted_product_ranking)
    | RunnablePassthrough.assign(
        ai_response4 = lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process fourth tool call
    | RunnablePassthrough.assign(
        tool_messages4 = lambda x: [
            execute_tool(tc) for tc in x["ai_response4"].tool_calls
        ]
    )
    # Final message update
    | RunnablePassthrough.assign(
        messages = lambda x: x["messages"]+ [x["ai_response4"]]+ x["tool_messages4"]
    )
    # Generate final summary
    | RunnablePassthrough.assign(
        summary = lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"])
)

if __name__ == "__main__":
    # Example use case:
    result = summarization_chain.invoke({"query": "Search for a wireless gaming mouse, I need it delivered by tomorrow."})
    print(result)
    