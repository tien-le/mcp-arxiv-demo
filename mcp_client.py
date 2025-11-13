# Chatbot POC
import asyncio

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# import nest_asyncio
# modifies the event loop so that nested loops can run safely inside Jupyter.
# nest_asyncio.apply()
# Running in Jupyter/IPython + using async code or libraries
# Prevents “event loop already running” error

# Applying Anthropic LLMs
# haiku : lighter & cheaper
# sonnet : mid-tier
# opus : highest tier

load_dotenv()

CHAT_LLM = "claude-3-haiku-20240307"
MAX_TOKENS = 100


#######################
# Chatbot POC
#######################


class MCP_ChatBot:
    """Chatbot POC using MCP."""

    def __init__(self):
        """Initialize the chatbot."""
        self.session: ClientSession = None
        self.client = anthropic.Anthropic()
        self.availables_tools: List[dict] = []

    # REVIEW: This is the main function that processes the user query and handles tool calls.
    async def process_query(self, query: str, model: str = CHAT_LLM) -> str:
        """
        Process a user query with the LLM and handle tool calls.

        Steps:
        1. Start the conversation with the user's message.
        2. Ask the model for a reply.
        3. If the reply is plain text, return it.
        4. If the reply asks to use tools, run all requested tools and send results back.
        5. Repeat until the model returns only text.
        """
        # Step 1: start conversation history
        messages = [{"role": "user", "content": query}]

        while True:
            print("*" * 30)

            # Step 2: get response from the model
            response = self.client.messages.create(
                max_tokens=MAX_TOKENS,
                model=model,
                tools=self.availables_tools,  # tools exposed to LLM
                messages=messages,
            )
            print(f"1-response: {response}")
            # 1-response: Message(id='msg_014YxmwjERR7c4eCshqtdj1u', content=[ToolUseBlock(id='toolu_01HxL689CsA23qtGcD9XLRt4', input={'topic': 'llm'}, name='search_papers', type='tool_use')], model='claude-3-haiku-20240307', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=Usage(cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0), cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=454, output_tokens=54, server_tool_use=None, service_tier='standard'))

            # Separate text and tool requests
            text_blocks = []
            tool_requests = []

            for block in response.content:
                print("-" * 30)
                print(f"2-block: {block}")
                # 2-block: ToolUseBlock(id='toolu_01HxL689CsA23qtGcD9XLRt4', input={'topic': 'llm'}, name='search_papers', type='tool_use')
                if block.type == "text":
                    text_blocks.append({"type": "text", "text": block.text})
                    print(f"3-text_blocks: {text_blocks}")
                elif block.type == "tool_use":
                    tool_requests.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input or {},
                        }
                    )
                    print(f"4-tool_requests: {tool_requests}")
                    # 4-tool_requests: [{'type': 'tool_use', 'id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'name': 'search_papers', 'input': {'topic': 'llm'}}]

            # Step 3: if only text, return it
            if text_blocks and not tool_requests:
                messages.append({"role": "assistant", "content": text_blocks})
                print(f"5-messages: {messages}")
                return "\n\n".join(block["text"] for block in text_blocks)

            # Step 4: handle tool requests
            if tool_requests:
                messages.append(
                    {"role": "assistant", "content": text_blocks + tool_requests}
                )
                print(f"6-messages: {messages}")
                # 6-messages: [{'role': 'user', 'content': 'what is llm?'}, {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'name': 'search_papers', 'input': {'topic': 'llm'}}]}]
                tool_results = []
                for tool in tool_requests:
                    print(f"7-tool: {tool}")
                    # 7-tool: {'type': 'tool_use', 'id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'name': 'search_papers', 'input': {'topic': 'llm'}}
                    # result = execute_tool(tool["name"], tool["input"])  # NO NEED ANYMORE
                    result = await self.session.call_tool(tool["name"], tool["input"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": result,
                        }
                    )
                    print(f"8-tool_results: {tool_results}")
                    # 8-tool_results: [{'type': 'tool_result', 'tool_use_id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'content': '2412.18022v1, 2406.10300v1, 2405.19888v1, 2311.10372v2, 2411.15764v1'}]
                messages.append({"role": "user", "content": tool_results})
                print(f"9-messages: {messages}")
                # 9-messages: [{'role': 'user', 'content': 'what is llm?'}, {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'name': 'search_papers', 'input': {'topic': 'llm'}}]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_01HxL689CsA23qtGcD9XLRt4', 'content': '2412.18022v1, 2406.10300v1, 2405.19888v1, 2311.10372v2, 2411.15764v1'}]}]
                continue

            # Fallback: if we reach here, return whatever text we have
            if text_blocks:
                print(f"10-text_blocks: {text_blocks}")
                return "\n\n".join(block["text"] for block in text_blocks)
            print("11-return empty string")
            return ""

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("Type your queries or quit/q/exit to exit.")
        while True:
            try:
                query = input("\nQuery:").strip()
                if query.lower() in ["q", "quit", "exit"]:
                    break
                answer = await self.process_query(query)
                if answer:
                    print(answer)
                print("\n")
            except Exception as e:
                print(f"Error while chatting, due to: {e}")

    async def launch_to_server(self):
        """Launch the chatbot to the server."""
        # Create server params for stdio connection
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "tien_mcp_server.py"],  # Optional command line arguments
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()

                # List available tools
                response = await session.list_tools()
                availables_tools = response.tools
                print("\nList of available tools:", [t.name for t in availables_tools])
                self.availables_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in availables_tools
                ]
                await self.chat_loop()


async def main():
    chat = MCP_ChatBot()
    await chat.launch_to_server()


if __name__ == "__main__":
    asyncio.run(main())

"""
(demo_mcp) mcp_project$ uv run tien_mcp_client.py
[11/11/25 15:17:50] INFO     Processing request of type ListToolsRequest                                                                       server.py:674

List of available tools: ['search_papers', 'extract_info']
Type your queries or quit/q/exit to exit.

Query:what is llm?
******************************
1-response: Message(id='msg_01DNXxU4LsuMDkK8n5PmnHjc', content=[TextBlock(citations=None, text='LLM stands for "Large Language Model". It refers to a type of AI model that is trained on a massive amount of text data to learn the patterns and structures of natural language. Some key characteristics of LLMs include:\n\n- They are trained on huge datasets, often containing billions of words from websites, books, articles, and other text sources.\n- They learn to understand and generate human-like text by identifying statistical patterns in this training data.\n- LLMs like', type='text')], model='claude-3-haiku-20240307', role='assistant', stop_reason='max_tokens', stop_sequence=None, type='message', usage=Usage(cache_creation=CacheCreation(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=0), cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=454, output_tokens=100, server_tool_use=None, service_tier='standard'))
------------------------------
2-block: TextBlock(citations=None, text='LLM stands for "Large Language Model". It refers to a type of AI model that is trained on a massive amount of text data to learn the patterns and structures of natural language. Some key characteristics of LLMs include:\n\n- They are trained on huge datasets, often containing billions of words from websites, books, articles, and other text sources.\n- They learn to understand and generate human-like text by identifying statistical patterns in this training data.\n- LLMs like', type='text')
3-text_blocks: [{'type': 'text', 'text': 'LLM stands for "Large Language Model". It refers to a type of AI model that is trained on a massive amount of text data to learn the patterns and structures of natural language. Some key characteristics of LLMs include:\n\n- They are trained on huge datasets, often containing billions of words from websites, books, articles, and other text sources.\n- They learn to understand and generate human-like text by identifying statistical patterns in this training data.\n- LLMs like'}]
5-messages: [{'role': 'user', 'content': 'what is llm?'}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'LLM stands for "Large Language Model". It refers to a type of AI model that is trained on a massive amount of text data to learn the patterns and structures of natural language. Some key characteristics of LLMs include:\n\n- They are trained on huge datasets, often containing billions of words from websites, books, articles, and other text sources.\n- They learn to understand and generate human-like text by identifying statistical patterns in this training data.\n- LLMs like'}]}]
LLM stands for "Large Language Model". It refers to a type of AI model that is trained on a massive amount of text data to learn the patterns and structures of natural language. Some key characteristics of LLMs include:

- They are trained on huge datasets, often containing billions of words from websites, books, articles, and other text sources.
- They learn to understand and generate human-like text by identifying statistical patterns in this training data.
- LLMs like



Query:q
"""
