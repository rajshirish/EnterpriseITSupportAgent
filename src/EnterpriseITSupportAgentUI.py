from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_classic.chains import RetrievalQA
import gradio as gr
from llm import model
from Knowledgebase import vector_store


user_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])
chat_history = []

USER_PROMPT_TEMPLATE = """Use the following pieces of the context to answer user's question.
If you don't know the answer, just say you don't know, don't try to make up the answer.
------------------------
{context}
Question: {question}
"""

USER_PROMPT = PromptTemplate(
    template=USER_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)
qa_chain = RetrievalQA.from_chain_type(
    model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": USER_PROMPT}
)


def chatbot_response(message, history):
    # Use the created RetrievalQA chain to get the answer
    response = qa_chain({"query": message})
    answer = response["result"]
    source_documents = response["source_documents"]

    # Format the response to include the answer and source documents (optional)
    formatted_response = f"{answer}" # You can add source documents here if desired

    print(f"Source documents: {source_documents}")

    history = [
            {"role": "assistant", "content": formatted_response}
        ]
    return history


# Function that gets called during every prompt submission
def get_response(user_input: str, history: str):
    global chat_history
    print(f"User input: {user_input}")
    user_prompt = user_prompt_template.format_messages(chat_history=chat_history, user_input=user_input)
    response = model.invoke(user_prompt)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(SystemMessage(content=response.content))
    print(response.content)
    #history.append(AIMessage(content=response.content))
    history = [
        {"role": "assistant", "content": response.content}
    ]
    return history

# Create Gradio Chat Interface

with gr.Blocks() as demo:
    gr.Markdown("## 💬 Chat with Enterprise IT Support Agent")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.Button("Clear Chat")

    msg.submit(chatbot_response, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# Run the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7868)

