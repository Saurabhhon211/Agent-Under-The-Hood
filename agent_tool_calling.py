from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langsmith import traceable

max_iterations = 10
model = "qwen3:1.7b"

@tool
def get_price(product:str) -> float:
    '''To get the price of the product'''
    print(f"Executing the tool get_price for the product::{product}")
    price_data = {"laptop" : 49000,
                  "mobile" : 20000, 
                  "headset": 2000}
    return price_data.get(product,0)


@tool
def get_discount(price:float,discount_tier: str) ->float:
    '''Apply discount tier to price and return the final price.
    Available Tiers : bronze, silver & gold'''
    print(f"Executing thetool get_dicount for the price::{price} and discount tier::{discount_tier}")
    discount_percentage = {"bronze" : 5,
                           "silver" : 10,
                           "gold"   : 15}

    discount = discount_percentage.get(discount_tier,0)
    return round(price * (1-discount/100),2)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_price,get_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{model}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question::::{question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are helpful shopping assistant."
                "You have access to product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - You must follow this exactly:\n"
                "1. Never guess r assume any product price. "
                "You must call get_price first to get the real price.\n"
                "2. Only call get_discount after you received the price from get_price."
                "Pass the exact price returned by get_price - do not pass a made-up number.\n"
                "3. Never calculate discount yourself using math."
                "Always use get_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do not assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    
    for iterations in range(1,max_iterations + 1):
        print(f"Iterations::{iterations}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer::{ai_message.content}")
            return ai_message.content

        messages.append(ai_message)
        for tool_call in tool_calls:
            tool = tools_dict.get(tool_call["name"])
            if tool is None:
                raise ValueError(f"Unknown tool requested: {tool_call['name']}")

            result = tool.invoke(tool_call["args"])
            print(f"{tool_call['name']} result::{result}")
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    raise RuntimeError(f"Agent exceeded the maximum of {max_iterations} iterations")

if __name__ == "__main__":
    print("Running the Langchain Agent::")
    print()
    result = run_agent("What is the price of laptop after applying gold discount?")