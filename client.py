import os
import json
import asyncio
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
from contextlib import AsyncExitStack

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    from anthropic._exceptions import APIError, RateLimitError, APIConnectionError
except ImportError:
    print("Please install the anthropic library: uv add anthropic")
    sys.exit(1)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool
except ImportError:
    print("Please install the MCP library: uv add mcp")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.text import Text
except ImportError:
    print("Please install rich: uv add rich")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv is optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    description: str = ""

@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime
    tool_calls: Optional[List[Dict[str, Any]]] = None

class MCPManager:
    """Manages MCP server connections using official MCP Python SDK"""
    
    def __init__(self, servers_config: List[MCPServerConfig]):
        self.servers_config = servers_config
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools: Dict[str, List[Tool]] = {}
    
    async def start_servers(self):
        """Start all configured MCP servers using official SDK"""
        for server_config in self.servers_config:
            try:
                # Build command and arguments
                args = [server_config.command]
                if server_config.args:
                    args.extend(server_config.args)
                
                # Set up environment
                env = os.environ.copy()
                if server_config.env:
                    env.update(server_config.env)
                
                # Create server parameters
                server_params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args or [],
                    env=env
                )
                
                # Connect using official MCP client
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                
                # Create session
                session = await self.exit_stack.enter_async_context(
                    ClientSession(*stdio_transport)
                )
                
                # Initialize the session
                await session.initialize()
                
                # Store the session
                self.sessions[server_config.name] = session
                
                # Get available tools using official SDK
                tools_response = await session.list_tools()
                self.available_tools[server_config.name] = tools_response.tools
                
                logger.info(f"Connected to MCP server '{server_config.name}' with {len(tools_response.tools)} tools")
                for tool in tools_response.tools:
                    logger.info(f"  - {tool.name}: {tool.description}")
                
            except Exception as e:
                logger.error(f"Failed to start MCP server '{server_config.name}': {e}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool using official MCP SDK"""
        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected")
        
        session = self.sessions[server_name]
        
        try:
            result = await session.call_tool(tool_name, arguments)
            
            if result.content and len(result.content) > 0:
                return result.content[0].text
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Tool call failed for {server_name}.{tool_name}: {e}")
            return f"Error: {str(e)}"
    
    def get_all_tools_for_anthropic(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic SDK format"""
        anthropic_tools = []
        
        for server_name, tools in self.available_tools.items():
            for tool in tools:
                anthropic_tool = {
                    "name": f"{server_name}__{tool.name}",
                    "description": f"[{server_name}] {tool.description or tool.name}",
                    "input_schema": tool.inputSchema or {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def get_available_tools(self) -> Dict[str, List[Tool]]:
        """Get all available MCP tools"""
        return self.available_tools.copy()
    
    async def cleanup(self):
        """Cleanup all MCP connections"""
        try:
            await self.exit_stack.close()
            logger.info("All MCP servers disconnected")
        except Exception as e:
            logger.error(f"Error during MCP cleanup: {e}")

class ClaudeChatbot:
    """Main chatbot class using official Anthropic SDK with MCP integration"""
    
    def __init__(self, api_key: str, mcp_servers: List[MCPServerConfig], max_context_messages: int = 20):
        self.client = AsyncAnthropic(api_key=api_key)
        self.mcp_manager = MCPManager(mcp_servers)
        self.max_context_messages = max_context_messages
        self.conversation_history: List[ChatMessage] = []
        self.session_file = Path("chat_session.json")
        self.console = Console()
        
    async def initialize(self):
        """Initialize the chatbot and MCP servers"""
        await self.mcp_manager.start_servers()
        await self.load_session()
        
    async def load_session(self):
        """Load conversation history from file"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    
                self.conversation_history = []
                for msg_data in data.get('messages', []):
                    msg = ChatMessage(
                        role=msg_data['role'],
                        content=msg_data['content'],
                        timestamp=datetime.fromisoformat(msg_data['timestamp']),
                        tool_calls=msg_data.get('tool_calls')
                    )
                    self.conversation_history.append(msg)
                    
                logger.info(f"Loaded {len(self.conversation_history)} messages from session")
            except Exception as e:
                logger.error(f"Failed to load session: {e}")
    
    async def save_session(self):
        """Save conversation history to file"""
        try:
            data = {
                'messages': []
            }
            
            for msg in self.conversation_history:
                msg_dict = asdict(msg)
                msg_dict['timestamp'] = msg.timestamp.isoformat()
                data['messages'].append(msg_dict)
            
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _prepare_messages_for_api(self) -> List[Dict[str, Any]]:
        """Prepare conversation history for Claude API"""
        recent_messages = self.conversation_history[-self.max_context_messages:]
        
        api_messages = []
        for msg in recent_messages:
            if msg.role in ["user", "assistant"]:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return api_messages
    
    async def _handle_tool_calls(self, message: Message) -> List[Dict[str, Any]]:
        """Handle tool calls from Claude using MCP SDK"""
        tool_results = []
        
        for content_block in message.content:
            print(content_block)
            if content_block.type == "tool_use":
                tool_name = content_block.name
                arguments = content_block.input
                tool_use_id = content_block.id
                
                # Parse server name from tool name
                if "__" in tool_name:
                    server_name, actual_tool_name = tool_name.split("__", 1)
                else:
                    logger.error(f"Invalid tool name format: {tool_name}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "Error: Invalid tool name format"
                    })
                    continue
                
                try:
                    # Call tool using MCP SDK
                    result = await self.mcp_manager.call_tool(server_name, actual_tool_name, arguments)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": str(result)
                    })
                    
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Error: {str(e)}"
                    })
        
        return tool_results
    
    async def send_message_stream(self, user_input: str):
        """Send a message to Claude and yield streaming response"""
        # Add user message to history
        user_msg = ChatMessage(
            role="user",
            content=user_input,
            timestamp=datetime.now()
        )
        self.conversation_history.append(user_msg)
        
        try:
            # Prepare messages and tools for API
            messages = self._prepare_messages_for_api()
            tools = self.mcp_manager.get_all_tools_for_anthropic()
            
            # Make streaming API call
            kwargs = {
                "model": "claude-3-7-sonnet-latest",
                "max_tokens": 1000,
                "messages": messages
            }
            
            if tools:
                kwargs["tools"] = tools
            
            assistant_content = ""
            tool_calls_made = []
            
            async with self.client.messages.stream(**kwargs) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        if chunk.delta.type == "text_delta":
                            text_chunk = chunk.delta.text
                            assistant_content += text_chunk
                            yield text_chunk
                    elif chunk.type == "content_block_start":
                        if chunk.content_block.type == "tool_use":
                            tool_calls_made.append({
                                "id": chunk.content_block.id,
                                "name": chunk.content_block.name,
                                "input": chunk.content_block.input
                            })
                
                # Get final message
                final_message = await stream.get_final_message()
            
            # Handle tool calls if present
            if final_message.stop_reason == "tool_use":
                tool_results = await self._handle_tool_calls(final_message)
                
                if tool_results:
                    yield "\n\n🔧 Executing tools...\n"
                    
                    # Prepare messages with tool results
                    tool_messages = messages + [
                        {"role": "assistant", "content": final_message.content},
                        {"role": "user", "content": tool_results}
                    ]
                    
                    # Make follow-up API call with tool results
                    async with self.client.messages.stream(
                        model="claude-3-7-sonnet-latest",
                        max_tokens=1000,
                        messages=tool_messages,
                        tools=tools if tools else []
                    ) as follow_up_stream:
                        async for chunk in follow_up_stream:
                            if chunk.type == "content_block_delta":
                                if chunk.delta.type == "text_delta":
                                    text_chunk = chunk.delta.text
                                    assistant_content += text_chunk
                                    yield text_chunk
            
            # Add final assistant message to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=assistant_content,
                timestamp=datetime.now(),
                tool_calls=tool_calls_made if tool_calls_made else None
            )
            self.conversation_history.append(assistant_msg)
            
            # Save session
            await self.save_session()
            
        except RateLimitError as e:
            yield f"\n⚠️ Rate limit exceeded. Please wait a moment and try again."
        except APIConnectionError as e:
            yield f"\n❌ Connection error: {e}"
        except APIError as e:
            yield f"\n❌ API error: {e}"
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            yield f"\n❌ Unexpected error: {e}"
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_manager.cleanup()
        await self.save_session()
        await self.client.aclose()

def load_config() -> Dict[str, Any]:
    """Load configuration from environment and config file"""
    config = {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "max_context_messages": int(os.getenv("MAX_CONTEXT_MESSAGES", "20")),
        "mcp_servers": []
    }
    
    # Load MCP servers from config file
    config_file = os.getenv("MCP_CONFIG_FILE", "mcp_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                mcp_config = json.load(f)
                
            for server_data in mcp_config.get("servers", []):
                server = MCPServerConfig(
                    name=server_data["name"],
                    command=server_data["command"],
                    args=server_data.get("args"),
                    env=server_data.get("env"),
                    description=server_data.get("description", "")
                )
                config["mcp_servers"].append(server)
                
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
    
    return config

async def main():
    """Main function to run the chatbot"""
    console = Console()
    
    # Load configuration
    config = load_config()
    
    if not config["api_key"]:
        console.print("[red]Please set ANTHROPIC_API_KEY environment variable[/red]")
        sys.exit(1)
    
    console.print(Panel.fit(
        "🤖 Claude Terminal Chatbot with MCP Integration\n"
        "Built with official Anthropic SDK and MCP Python SDK",
        title="Claude Chatbot",
        border_style="blue"
    ))
    
    console.print("\n[bold]Commands:[/bold]")
    console.print("  [cyan]/help[/cyan]     - Show this help")
    console.print("  [cyan]/clear[/cyan]    - Clear conversation history")
    console.print("  [cyan]/tools[/cyan]    - Show available MCP tools")
    console.print("  [cyan]/quit[/cyan]     - Exit the chatbot")
    console.print("  [cyan]/stats[/cyan]    - Show conversation statistics")
    
    # Initialize chatbot
    chatbot = ClaudeChatbot(
        api_key=config["api_key"],
        mcp_servers=config["mcp_servers"],
        max_context_messages=config["max_context_messages"]
    )
    
    try:
        with console.status("[bold green]Initializing chatbot and MCP servers..."):
            await chatbot.initialize()
        
        # Show available tools
        available_tools = chatbot.mcp_manager.get_available_tools()
        if available_tools:
            total_tools = sum(len(tools) for tools in available_tools.values())
            console.print(f"\n✅ Connected to [bold]{len(available_tools)}[/bold] MCP server(s) with [bold]{total_tools}[/bold] total tools:")
            for server_name, tools in available_tools.items():
                console.print(f"  • [bold]{server_name}[/bold]: {len(tools)} tools")
        else:
            console.print("\n⚠️  No MCP servers configured or available")
        
        console.print("\n[green]Ready! Type your message and press Enter. Use /quit to exit.[/green]\n")
        
        # Main chat loop
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/quit':
                        break
                    elif command == '/help':
                        console.print("\n[bold]Commands:[/bold]")
                        console.print("  [cyan]/help[/cyan]     - Show this help")
                        console.print("  [cyan]/clear[/cyan]    - Clear conversation history")
                        console.print("  [cyan]/tools[/cyan]    - Show available MCP tools")
                        console.print("  [cyan]/quit[/cyan]     - Exit the chatbot")
                        console.print("  [cyan]/stats[/cyan]    - Show conversation statistics\n")
                        continue
                    elif command == '/clear':
                        chatbot.conversation_history.clear()
                        await chatbot.save_session()
                        console.print("[yellow]🗑️  Conversation history cleared.[/yellow]\n")
                        continue
                    elif command == '/tools':
                        tools = chatbot.mcp_manager.get_available_tools()
                        if tools:
                            console.print("\n[bold blue]🔧 Available MCP Tools:[/bold blue]")
                            for server_name, server_tools in tools.items():
                                console.print(f"\n  [bold]📦 {server_name}:[/bold]")
                                for tool in server_tools:
                                    desc = tool.description or 'No description'
                                    console.print(f"    • [cyan]{tool.name}[/cyan]: {desc}")
                        else:
                            console.print("[yellow]No MCP tools available.[/yellow]")
                        console.print()
                        continue
                    elif command == '/stats':
                        total_messages = len(chatbot.conversation_history)
                        user_messages = len([m for m in chatbot.conversation_history if m.role == "user"])
                        assistant_messages = len([m for m in chatbot.conversation_history if m.role == "assistant"])
                        
                        console.print(f"\n[bold]📊 Conversation Statistics:[/bold]")
                        console.print(f"  Total messages: [cyan]{total_messages}[/cyan]")
                        console.print(f"  User messages: [green]{user_messages}[/green]")
                        console.print(f"  Assistant messages: [blue]{assistant_messages}[/blue]")
                        console.print(f"  Context window: [yellow]{config['max_context_messages']} messages[/yellow]")
                        console.print(f"  MCP servers: [magenta]{len(available_tools)}[/magenta]\n")
                        continue
                    elif command == '/logs':
                        try:
                            if Path('mcp_requests.log').exists():
                                with open('mcp_requests.log', 'r') as f:
                                    lines = f.readlines()
                                    # Show last 50 lines
                                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                                    
                                console.print("\n[bold blue]📋 Recent MCP Request Logs (last 50 lines):[/bold blue]")
                                console.print("[dim]" + "".join(recent_lines) + "[/dim]")
                                console.print(f"\n[yellow]Full logs available in: mcp_requests.log[/yellow]\n")
                            else:
                                console.print("[yellow]No MCP request logs found yet.[/yellow]\n")
                        except Exception as e:
                            console.print(f"[red]Error reading logs: {e}[/red]\n")
                        continue
                    else:
                        console.print(f"[red]Unknown command: {user_input}[/red]")
                        continue
                
                # Send message to Claude with streaming
                console.print("[bold blue]Claude:[/bold blue] ", end="")
                
                async for chunk in chatbot.send_message_stream(user_input):
                    console.print(chunk, end="", highlight=False)
                
                console.print()  # New line after response
                console.print()  # Extra spacing
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                console.print(f"[red]Error: {e}[/red]\n")
    
    finally:
        with console.status("[bold yellow]Cleaning up..."):
            await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())