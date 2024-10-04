import streamlit as st
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage



# Define your tools
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b

def add(a: int, b: int) -> int:
    """
    Adds two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b

def divide(a: int, b: int) -> float:
    """
    Divides the first integer by the second integer.

    Args:
        a (int): The numerator.
        b (int): The denominator.

    Returns:
        float: The result of the division.
    """
    return a / b

tools = [add, multiply, divide]

# Sidebar for API key input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your API key:", type="password")

if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
   

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    # Node
    def assistant(state: MessagesState):
        """
        Invokes the language model with the given state.

        Args:
            state (MessagesState): The current state of messages.

        Returns:
            dict: A dictionary containing the response messages.
        """
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    react_graph = builder.compile()

    st.title("LangGraph with Streamlit")
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            role = "user"
        if isinstance(message, AIMessage):
            role = "assistant"
        
        with st.chat_message(role):
            st.markdown(message.content)
        # if isinstance(message, AIMessage):
        #     st.markdown(message.content)

    if prompt := st.chat_input("What is up?"):
        # messages = [HumanMessage(content=prompt)]
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
              st.markdown(prompt)

        result = react_graph.invoke({"messages": st.session_state.messages})
        with st.chat_message("assistant"):
            # print(result)
            m = result['messages'][-1]
            if isinstance(m, AIMessage):
                st.markdown(m.content)
                st.session_state.messages.append(m)
else:
    st.warning("Please enter your API key to enable the chat.")
