# A pydantic Model for data validation
from typing import Dict, Any, List, Literal
from langchain_core.messages import(
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
    message_to_dict,
    messages_from_dict,
)
from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Basic user input for the agent."""
    
    messages: str = Field(
        description="User input to the agent.",
        examples=["What does Lilian Weng say about types of reward hacking?"],
    )
    
    model: str = Field(
        description="LLM Model to use for the agent.",
        default="gpt-5-nano",
        examples=["gpt-4o-mini", "llama-3.1-70b"],
    )
    
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    

class StreamInput(UserInput):
    """User input for streaming the agent's response."""
    
    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )
    

class AgentResponse(BaseModel):
    """Response from the agent when called via /invoke."""
    
    message: Dict[str, Any] = Field(
        description="Final response from the agent, as a serialized LangChain message.",
        examples=[
            {
                "message": {
                    "type": "ai",
                    "data": {"content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering.", "type": "ai"},
                }
            }
        ],
    )

# Pydantic for Model -> Database -> Model  
class ChatMessage(BaseModel):
    """Message in a chat."""
    
    type: Literal["human", "ai", "tool"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool"],
    )
    
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    
    tool_calls: List[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    
    original: Dict[str, Any] = Field(
        description="Original LangChain message in serialized form.",
        default={},
    )
    
    @classmethod
    def from_langchain(cls, message:BaseMessage) -> "ChatMessage":  # This function specifies after the agent has finished it's job, it looks like python object which cannot be saved into postgresql db or connect with react frontend, we do need JSON objects pydantic helps convert this python object from langchain to json object. # LangChain Object --->> Pydantic/JSON.
        """Create a ChatMessage from a LangChain message."""
        original = message_to_dict(message)  # Turns the object into a Python Dictionary (JSON-like) Now it looks like this (Simplified JSON):  {"type": "ai", "content": "Reward hacking is...", "tool_calls": [...], "original": { ... a massive dictionary of all metadata ... }}
        match message:
            case HumanMessage():
                human_message = cls(type="human", content=message.content, original=original)
                return human_message
            case AIMessage():
                ai_message = cls(type="ai", content=message.content, original=original)
                if message.tool_calls:
                    ai_message.tool_calls = message.tool_calls
                return ai_message
            case ToolMessage():
                tool_message = cls(
                    type="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    original=original,
                )
                return tool_message
            case _:
                raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
            
    # after from_langchain function the text is just a string a json object which cannot be understood by any user but databases.
            
    def to_langchain(self) -> None:   # the database gives us a static json which cannot be presented to a agent/llm. It's used when we are loading history from our database back into the AI's brain. # JSON/Pydantic ---->> LangChain Object.
        """Convert the ChatMessage to a LangChain message."""
        if self.original:
            return messages_from_dict([self.original])[0] # This returns a REAL AIMessage object again!
        # If we don't have the original snapshot we rebuild it.
        match self.type:
            case "human":
                return HumanMessage(content=self.content)
            case "ai":
                return AIMessage(content=self.content, tool_calls=self.tool_calls)
            case "tool":
                return ToolMessage(content=self.content, tool_call_id=self.tool_call_id)
            case _:
                raise NotImplementedError(f"Unsupported message type: {self.type}")
            
    
    def pretty_print(self) -> None:
        """Pretty print the ChatMessage."""
        lc_msg = self.to_langchain()
        lc_msg.pretty_print()
                
            
            
# Edge Case : 1. What if the user keeps on asking the same queries multiple times then the tokens will increase?