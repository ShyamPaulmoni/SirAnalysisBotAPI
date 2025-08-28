from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import uuid
from datetime import datetime
from utils import (
    detect_english_names, convert_english_to_tamil_names, 
    get_conversation_context, generate_sql_query, execute_sql_query,
    explain_results, generate_follow_up_suggestions
)

# Initialize FastAPI app
app = FastAPI(
    title="Tamil Voter Data Chatbot API",
    description="AI-powered chatbot for Tamil voter data analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sql_query: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    results_count: int = 0
    explanation: Optional[str] = None
    follow_up_suggestions: List[str] = []
    name_conversion_info: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: str
    last_activity: str

# Global conversation storage (use Redis/Database in production)
conversations: Dict[str, List[Dict[str, Any]]] = {}
session_metadata: Dict[str, Dict[str, str]] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "healthy", "message": "Tamil Voter Data Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - handles all chatbot interactions"""
    try:
        # Handle session
        session_id = request.session_id
        if not session_id or session_id not in conversations:
            session_id = str(uuid.uuid4())
            conversations[session_id] = []
            session_metadata[session_id] = {
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
        
        # Update last activity
        session_metadata[session_id]["last_activity"] = datetime.now().isoformat()
        
        # Get conversation history
        conversation_history = conversations[session_id]
        
        # Add user message to history
        user_message = {
            "role": "user", 
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }
        conversations[session_id].append(user_message)
        
        # Process the message internally
        response_data = await _process_chat_message(request.message, conversation_history)
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.now().isoformat()
        }
        conversations[session_id].append(assistant_message)
        
        # Return response
        return ChatResponse(
            response=response_data["response"],
            session_id=session_id,
            sql_query=response_data.get("sql_query"),
            results=response_data.get("results"),
            results_count=response_data.get("results_count", 0),
            explanation=response_data.get("explanation"),
            follow_up_suggestions=response_data.get("follow_up_suggestions", []),
            name_conversion_info=response_data.get("name_conversion_info")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

async def _process_chat_message(message: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Internal function to process chat messages and handle all operations"""
    
    # Check for English name conversion (but skip if constituency names are detected)
    name_conversion_info = None
    tamil_names = []
    
    # Don't convert names if constituency names are detected
    if detect_english_names(message):
        conversion_result = convert_english_to_tamil_names(message)
        if conversion_result["tamil_names"]:
            tamil_names = conversion_result["tamil_names"]
            mapping_text = " | ".join([f"{eng} → {tam}" for eng, tam in conversion_result["name_mapping"].items()])
            name_conversion_info = f"✅ Name conversion: {mapping_text}"
    
    # Get conversation context
    context = get_conversation_context(conversation_history)
    
    # Generate SQL query based on user input
    sql_query = generate_sql_query(
        question=message,
        tamil_names=tamil_names if tamil_names else None,
        context=context
    )
    
    # Execute query
    try:
        results = execute_sql_query(sql_query)
        results_count = len(results)
        
        if results_count > 0:
            # Generate explanation
            explanation = explain_results(
                question=message,
                query=sql_query,
                results=results,
                used_name_conversion=bool(tamil_names),
                context=context
            )
            
            # Generate follow-up suggestions
            follow_ups = generate_follow_up_suggestions(
                question=message,
                query=sql_query,
                results=results,
                context=context
            )
            
            # Create response message
            response_text = f"Found {results_count} records."
            if name_conversion_info:
                response_text += f"\n\n{name_conversion_info}"
            response_text += f"\n\n**Analysis:** {explanation}"
            
            return {
                "response": response_text,
                "sql_query": sql_query,
                "results": results,
                "results_count": results_count,
                "explanation": explanation,
                "follow_up_suggestions": follow_ups,
                "name_conversion_info": name_conversion_info
            }
        else:
            response_text = "No results found for your query."
            if name_conversion_info:
                response_text += f"\n\n{name_conversion_info}"
            response_text += "\n\nTry rephrasing your question or ask about general demographics."
            
            return {
                "response": response_text,
                "sql_query": sql_query,
                "results": [],
                "results_count": 0,
                "follow_up_suggestions": [
                    "Show me overall demographics",
                    "What are the age distributions?", 
                    "How many total voters are there?"
                ],
                "name_conversion_info": name_conversion_info
            }
            
    except Exception as e:
        return {
            "response": f"I encountered an error processing your query: {str(e)}. Please try rephrasing your question.",
            "sql_query": sql_query,
            "results": [],
            "results_count": 0,
            "follow_up_suggestions": [
                "Show basic voter statistics",
                "How many male and female voters?",
                "What's the age distribution?"
            ]
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )