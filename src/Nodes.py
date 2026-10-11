# STEP 1 - Define the central state of the application
import Tools
from llm import model
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from Tools import ALL_TOOLS
from langgraph.graph import StateGraph, START, END
from langchain.agents.middleware import PIIMiddleware

# State of the Blog Graph
from typing import TypedDict

class EntITSupportState(TypedDict):
    InputValidation: str
    RequestClassification: str
    Retrieval: str
    DocumentGrading: str
    AgentSelection: str
    ToolExecution: str
    HumanApproval: str
    ResponseGeneration: str
    OutputValidation: str 

print('Enterprise IT Support State is ready')


# STEP 2 - Define your nodes

llm_with_tools = model.bind_tools(ALL_TOOLS)

# Node definition

# Node 1
def InputValidation_node(state: EntITSupportState):
    # write code to validate the input - need to pass at least 10 constraints for a valid input
    # 50 lines of code
    # Create agent with PII Middleware
    agent = create_agent(
        model=model,
        tools=[Tools.check_employee_account],
        middleware=[
            # Redact emails in user input before sending to model
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
            ),
            # Mask credit cards in user input
            PIIMiddleware(
                "credit_card",
                strategy="mask",
                apply_to_input=True,
            ),
            # Block API keys - raise error if detected
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="block",
                apply_to_input=True,
            ),
        ],
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": state['InputValidation']
        }]
    })

    res = result["messages"][-1].content

    return state

# Node 2
def RequestClassification_node(state: EntITSupportState):
    topic = state['topic']
    #agent.invoke(f"Please write 5 lines of introduction about {topic}")
    return {
        "introduction": f"This is an introduction about {topic}"
    }

# Node 3
def Retrieval_node(state: EntITSupportState):
    topic = state['topic']
    return {
        "examples": f"Please find below the examples on the topic {topic}"
    }

# Node 4
def DocumentGrading_node(state: EntITSupportState):
    topic = state['topic']
    return {
        "codes": f"Here are the code snippets on the topic {topic}"
    }

# Node 5
def AgentSelection_node(state: EntITSupportState):
    final_blog = (
        state['topic'] + '\n' +
        state['introduction'] + '\n' +
        state['examples'] + '\n' +
        state['codes'] + '\n'
    )
    return {
        "final_blog": final_blog
    }

def ToolExecution_node(state: EntITSupportState):
    final_blog = state['final_blog']
    return {
        "final_blog": final_blog
    }

def HumanApproval_node(state: EntITSupportState):
    final_blog = state['final_blog']
    return {
        "final_blog": final_blog
    }

def ResponseGeneration_node(state: EntITSupportState):
    final_blog = state['final_blog']
    return {
        "final_blog": final_blog
    }

def OutputValidation_node(state: EntITSupportState):
    final_blog = state['final_blog']
    return {
        "final_blog": final_blog
    }


def model_based_classification(text: str) -> str:
    """Uses an LLM to evaluate content safety and classify the request into three categories FAQ, TOOL, or UNSAFE. Returns the category FAQ, TOOL, or UNSAFE."""

    prompt = f"""Classify the following user input among these categories FAQ, TOOL, or UNSAFE?
    Reply with only 'FAQ', 'TOOL', or 'UNSAFE'.

Input: {text}"""
    result = model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()


class RequestClassification(TypedDict):
    Request: str
    Category: str

def classifyRequest(state: RequestClassification) -> dict:
    input = model_based_classification(state['Request'])
    state['Category'] = input
    return state

topic_classification_graph_builder = StateGraph(RequestClassification)

# Node registration
topic_classification_graph_builder.add_node("classifyRequest", classifyRequest)

# Add edges to nodes
topic_classification_graph_builder.add_edge(START, "classifyRequest")
topic_classification_graph_builder.add_edge("classifyRequest", END)

topic_classification_graph = topic_classification_graph_builder.compile()
print('Topic classification graph is ready')
topic_classification_graph

print('All nodes are ready')

EntITSupport_graph_builder = StateGraph(EntITSupportState)

# Node registration
EntITSupport_graph_builder.add_node("inputValidation", InputValidation_node)
EntITSupport_graph_builder.add_node("RequestClassification", RequestClassification_node)
EntITSupport_graph_builder.add_node("Retrieval", Retrieval_node)
EntITSupport_graph_builder.add_node("DocumentGrading ", DocumentGrading_node)
EntITSupport_graph_builder.add_node("AgentSelection", AgentSelection_node)
EntITSupport_graph_builder.add_node("ToolExecution ", ToolExecution_node)
EntITSupport_graph_builder.add_node("HumanApproval", HumanApproval_node)
EntITSupport_graph_builder.add_node("ResponseGeneration", ResponseGeneration_node)
EntITSupport_graph_builder.add_node("OutputValidation", OutputValidation_node)

# Every graph can be registered as a valid Node in another graph

# Add edges to nodes
EntITSupport_graph_builder.add_edge(START, "inputValidation")

EntITSupport_graph_builder.add_conditional_edges(
    "RequestClassification",
    lambda _state: "Retrieval" if _state['Retrieval'] else END,
    {
        "Retrieval": "Retrieval",
        END: END
    }
)

EntITSupport_graph_builder.add_edge("inputValidation", "RequestClassification")
EntITSupport_graph_builder.add_edge("RequestClassification", "Retrieval")
EntITSupport_graph_builder.add_edge("RequestClassification", "codes")

EntITSupport_graph_builder.add_edge("introduction", "merger")
EntITSupport_graph_builder.add_edge("examples", "merger")
EntITSupport_graph_builder.add_edge("codes", "merger")

EntITSupport_graph_builder.add_edge("merger", END)

EntITSupport_graph = EntITSupport_graph_builder.compile()
print('Enterprise IT Support graph is ready')
EntITSupport_graph