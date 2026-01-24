import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, ToolMessage
from amazon_tools import tools, tool_mapping

# 1. LLM Setup via Hugging Face Router
# Ensure you use your latest API Key here
os.environ["OPENAI_API_KEY"] = "hf_YOUR_NEW_TOKEN" 

llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.1
)

# Binding the tools to the LLM to enable tool usage capabilities
llm_with_tools = llm.bind_tools(tools)

# 2. Tool Execution Logic
def execute_tool(tool_call):
    """
    Finds the requested tool in the mapping and executes it with provided arguments.
    Returns a ToolMessage which is required for the conversation history.
    """
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    target_tool = tool_mapping[tool_name]
    output = target_tool.invoke(tool_args)
    
    return ToolMessage(content=str(output), tool_call_id=tool_call["id"])

# 3. LCEL Chain Construction
chain = (
    # STEP 1: Initialize the message history with user query
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    # STEP 2: First LLM call to decide which tool to use (usually URL generation)
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # STEP 3: Execute the tool(s) requested by the LLM
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [execute_tool(tc) for tc in x["ai_response"].tool_calls]
    )
    # STEP 4: Update conversation history with AI's thought and Tool's output
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
)

if __name__ == "__main__":
    # Example usage:
    # result = chain.invoke({"query": "Search for wireless gaming mouse on Amazon."})
    # print(result)
    print("Chain structure is ready. You can uncomment the invoke line to test.")