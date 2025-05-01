import threading
import subprocess
from typing import List, Any
import queue
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from conversation_stream import stream_cpp_output


class State(MessagesState):
    transcription_role: List[Any]
    transcription_content: List[Any]
    summary: str
    print_transcription = True
    summary_timepoint = None
    summary_pointer = None


if __name__ == '__main__':
    # Initialize a thread-safe queue
    output_queue = queue.Queue()

    # cmd_out = ["./build/bin/Release/whisper-stream.exe", "-m", "./models/ggml-base.en.bin",
    #         "-t", "6", "--step", "0", "--length", "3000", "-vth", "0.8", "-ac", "0", "-c", "1", "--step", "1000"]
    # cmd_in = ["./build/bin/Release/whisper-stream.exe", "-m", "./models/ggml-base.en.bin",
    #         "-t", "6", "--step", "0", "--length", "3000", "-vth", "0.8", "-ac", "0", "--step", "1000"]

    # # Create and start threads for each subprocess
    # thread1 = threading.Thread(
    #     target=stream_cpp_output, args=(cmd_out, "Output", output_queue))
    # thread2 = threading.Thread(
    #     target=stream_cpp_output, args=(cmd_in, "Input", output_queue))

    # thread1.start()
    # thread2.start()

    # # Process output in real-time
    # while thread1.is_alive() or thread2.is_alive() or not output_queue.empty():
    #     try:
    #         label, line = output_queue.get()
    #         # Disable SDL2 init log
    #         if line[:5] == "init:":
    #             continue
    #         print(f"[{label}] {line}")
    #     except queue.Empty:
    #         continue

    # # Ensure all threads have completed
    # thread1.join()
    # thread2.join()

    # prompt = "This is a foobar thing"
    inference_server_url = "http://bime-bragi.bime.washington.edu:5000/api"
    token = 'sk-cf37afff34904ad99e3a4e55e7731a02'
    llm = ChatOpenAI(
        model="llama3.3:latest",
        openai_api_key=token,
        openai_api_base=inference_server_url,
        max_tokens=5,
        temperature=0,
    )

    # Node
    # Define a node to maintain the live transcription
    def transcription_maintainer(state: State):
        top_k = 2
        cnt = 0  # no. of retrieved transcription
        roles = state["transcription_role"]
        content = state["transcription_content"]

        # Get top k elements from the queue
        while not output_queue.empty() and cnt < top_k:
            try:
                label, line = output_queue.get()
                # Disable SDL2 init log
                if line[:5] == "init:":
                    continue
                cnt += 1
                if state['print_transcription']:
                    print(f"[{label}] {line}")
                roles += label
                content += line
            except queue.Empty:
                continue
        return {"transcription_role": roles, "transcription_content": content}

    # Define the logic to call the model

    def feedback_checker(state: State):

        # TODO
        # Get summary if it exists
        summary = state.get("summary", "")

        # If there is summary, then we add it
        if summary:

            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(
                content=system_message)] + state["messages"]

        else:
            messages = state["messages"]

        response = llm.invoke(messages)
        return {"messages": response}

    def summarize_conversation(state: State):
        """
        Summarize past conversation
        """
        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt
        if summary:

            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )

        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = llm.invoke(messages)

        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id)
                           for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def summarize_condition(state: State):
        """
        Condition edge between transcription_maintainer and summarize_conversation
        """

        # TODO
        if len(messages) > 6:
            return "summarize_conversation"

        return "transcription_maintainer"

    def feedback_provider(state: State):
        # TODO
        # Get summary if it exists
        summary = state.get("summary", "")

        # If there is summary, then we add it
        if summary:

            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(
                content=system_message)] + state["messages"]

        else:
            messages = state["messages"]

        response = llm.invoke(messages)
        return {"messages": response}

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node(transcription_maintainer)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, transcription_maintainer)
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
