import threading
import duckdb
import pandas as pd
import re
from langchain_openai import AzureChatOpenAI
import os
import json
from typing import List, Dict, Optional, Any
from functools import lru_cache
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Global variables for caching
_llm_client = None
_db_connection = None
_schema_info = None
_sas_url = None
_sas_expiry = None
_sas_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_llm_client():
    """Initialize and return the Azure OpenAI client (cached)"""
    global _llm_client
    if _llm_client is None:
        try:
            api_key = os.getenv("API_KEY")
            api_base = os.getenv("API_BASE")
            deployment_name = os.getenv("DEPLOYMENT_NAME")
            
            if not api_key or "dummy" in api_key.lower():
                raise ValueError("Azure OpenAI API key not properly configured")
            if not api_base or "dummy" in api_base.lower():
                raise ValueError("Azure OpenAI API base URL not properly configured")
            if not deployment_name:
                raise ValueError("Azure OpenAI deployment name not configured")
                
            _llm_client = AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=api_base,
                openai_api_key=api_key,
                openai_api_type=os.getenv("API_TYPE", "azure"),
                openai_api_version=os.getenv("API_VERSION", "2025-01-01-preview"),
                temperature=0.1,
            )
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {str(e)}")
            raise
    return _llm_client


@lru_cache(maxsize=1)
def generate_sas_url_with_cache():
    """Generate SAS URL with intelligent caching and refresh"""
    global _sas_url, _sas_expiry
    
    with _sas_lock:
        # Check if we need to refresh (refresh 10 minutes before expiry)
        if _sas_url is None or _sas_expiry is None or datetime.now() > (_sas_expiry - timedelta(minutes=10)):
            try:
                connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if not connection_string or "dummy" in connection_string.lower():
                    raise ValueError("Azure Storage connection string not properly configured")
                    
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client("data")
                blob_client = container_client.get_blob_client("voter_duckdb/updated_merged.parquet")

                # Generate 2-hour token (refresh every 1h50m)
                expiry_time = datetime.now() + timedelta(hours=2)
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_client.container_name,
                    blob_name=blob_client.blob_name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=expiry_time,
                )

                _sas_url = f"{blob_client.url}?{sas_token}"
                _sas_expiry = expiry_time
                print(f"SAS URL refreshed. Expires at: {_sas_expiry}")
                
            except Exception as e:
                print(f"Error generating SAS URL: {str(e)}")
                raise
                
        return _sas_url

@lru_cache(maxsize=1)
def init_database():
    """Initialize DuckDB connection with proper container environment setup"""
    global _db_connection
    if _db_connection is None or _sas_expiry is None or datetime.now() > (_sas_expiry - timedelta(minutes=10)):
        try:
            # Create a writable directory in the container
            os.makedirs('/app/duckdb_data', exist_ok=True)
            
            # Initialize DuckDB with explicit configuration for containers
            _db_connection = duckdb.connect()
            
            # Set multiple fallback home directories for container compatibility
            _db_connection.execute("SET home_directory='/app/duckdb_data';")
            _db_connection.execute("SET temp_directory='/app/duckdb_data';")
            
            # Install and load httpfs extension
            _db_connection.execute("INSTALL httpfs;")
            _db_connection.execute("LOAD httpfs;")
            
            # Configure httpfs for Azure Blob Storage
            _db_connection.execute("SET s3_region='us-east-1';")
            _db_connection.execute("SET s3_use_ssl=true;")
            
            # Get fresh SAS URL and create view
            sas_url = generate_sas_url_with_cache()
            _db_connection.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet('{sas_url}')")
            
            print("DuckDB initialized successfully for container environment")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            # Fallback: try in-memory database
            try:
                _db_connection = duckdb.connect(':memory:')
                _db_connection.execute("INSTALL httpfs;")
                _db_connection.execute("LOAD httpfs;")
                sas_url = generate_sas_url_with_cache()
                _db_connection.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet('{sas_url}')")
                print("DuckDB initialized with in-memory fallback")
            except Exception as fallback_error:
                print(f"Both database initialization methods failed: {str(fallback_error)}")
                raise

    return _db_connection

@lru_cache(maxsize=1)
def get_schema_info():
    """Get database schema information and statistics (cached)"""
    global _schema_info
    if _schema_info is None:
        conn = init_database()
        schema = conn.execute("DESCRIBE data").df()
        sample = conn.execute("SELECT * FROM data LIMIT 5").df()
        stats = conn.execute("SELECT COUNT(*) as total_rows FROM data").df()
        
        # Convert DataFrames to dictionaries for JSON serialization
        _schema_info = {
            "schema": schema.to_dict('records'),
            "sample": sample.to_dict('records'),
            "stats": stats.to_dict('records')[0]
        }
    return _schema_info

def extract_person_names_with_ai(text: str) -> List[str]:
    """Use AI to identify person names in the text via Named Entity Recognition"""
    try:
        client = get_llm_client()
        prompt = f"""
         Analyze this text and extract ONLY person names (not places, organizations, or common words).
        
        Text: "{text}"
        
        TASK: Identify person names that could be Indian/Tamil names written in English.
        
        GUIDELINES:
        1. Look for proper nouns that could be person names
        2. Focus on Indian/South Indian name patterns
        3. Ignore place names, organizations, common English words
        4. Consider both first names and surnames
        5. Include names even if they appear in different contexts
        
        COMMON INDIAN NAME PATTERNS:
        - Names ending in: -an, -ar, -am, -raj, -kumar, -devi, -priya, -krishna
        - Names starting with: Sri, Sel, Muru, Kart, Div, Ani, Sur, Rav, Prab
        
        OUTPUT FORMAT:
        Return ONLY the person names found, one per line.
        If no person names found, return "NO_NAMES_FOUND"
        
        Examples:
        Input: "Find voters named Kumar and Priya in Chennai"
        Output: 
        Kumar
        Priya
        
        Input: "Show me demographics data"
        Output: NO_NAMES_FOUND
        """
        
        response = client.invoke(prompt)
        result = response.content.strip()
        
        if result == "NO_NAMES_FOUND":
            return []
        
        # Parse the response to extract names
        names = []
        for line in result.split('\n'):
            name = line.strip()
            if name and len(name) > 1 and not name.startswith('#'):
                # Additional validation - check if it looks like a name
                if re.match(r'^[A-Za-z]{2,}$', name) and name[0].isupper():
                    names.append(name)
        
        return names
        
    except Exception as e:
        print(f"Name extraction failed: {str(e)}")
        return []

def detect_english_names(text: str) -> bool:
    """Enhanced detection using AI-based Named Entity Recognition for person names only"""
     
    # First, try AI-based name extraction
    extracted_names = extract_person_names_with_ai(text)
    
    if extracted_names:
        return True
    
    # Fallback to pattern-based detection for common person names
    common_tamil_names = [
        'kumar', 'raj', 'priya', 'ravi', 'devi', 'krishna', 'lakshmi',
        'murugan', 'selvam', 'mani', 'karthik', 'divya', 'anita', 'suresh',
        'muthu', 'arjun', 'kavitha', 'deepa', 'ganesh', 'shiva',
        'saravanan', 'vijay', 'ajith', 'madhavi', 'radha', 'sita', 'gita'
    ]
    
    text_lower = text.lower()
    
    # Check for known Tamil person names in English
    for name in common_tamil_names:
        if name in text_lower:
            return True
    
    return False

def convert_english_to_tamil_names(english_text: str) -> Dict[str, Any]:
    """Convert English names to Tamil using OpenAI with extracted names"""
    try:
        client = get_llm_client()
        # First extract person names from the text
        extracted_names = extract_person_names_with_ai(english_text)
        
        if not extracted_names:
            return {"tamil_names": [], "name_mapping": {}}
        
        # Convert each extracted name to Tamil
        names_to_convert = ', '.join(extracted_names)
        
        prompt = f"""
        Convert these English person names to their Tamil script equivalents:
        
        Names to convert: {names_to_convert}
        
        IMPORTANT GUIDELINES:
        1. Convert each name to Tamil script
        2. Focus on South Indian/Tamil name conventions
        3. If a name has multiple Tamil variants, use the most common one
        4. Return only the Tamil equivalents, no explanations
        
        REFERENCE CONVERSIONS:
        Kumar → குமார்
        Raj → ராஜ்
        Priya → பிரியா  
        Ravi → ரவி
        Devi → தேவி
        Krishna → கிருஷ்ணா
        Lakshmi → லக்ஷ்மி
        Murugan → முருகன்
        Selvam → செல்வம்
        Mani → மணி
        Karthik → கார்த்திக்
        Divya → திவ்யா
        Anita → அனிதா
        Suresh → சுரேஷ்
        Muthu → முத்து
        Arjun → அர்ஜுன்
        Kavitha → கவிதா
        Deepa → தீபா
        Ganesh → கணேஷ்
        Shiva → சிவா
        Saravanan → சரவணன்
        Vijay → விஜய்
        Ajith → அஜித்
        Madhavi → மாதவி
        Radha → ராதா
        Sita → சீதா
        Gita → கீதா
        
        OUTPUT FORMAT:
        Return the Tamil names separated by commas.
        If a name cannot be converted, skip it.
        
        Example:
        Input: Kumar, Priya
        Output: குமார், பிரியா
        """
        
        response = client.invoke(prompt)
        converted_text = response.content.strip()
        
        # Parse the response to extract Tamil names
        tamil_names = [name.strip() for name in converted_text.split(',') if name.strip()]
        
        # Create the mapping
        name_mapping = {}
        if tamil_names and len(tamil_names) == len(extracted_names):
            for i, eng_name in enumerate(extracted_names):
                if i < len(tamil_names):
                    name_mapping[eng_name] = tamil_names[i]
        
        return {
            "tamil_names": tamil_names,
            "name_mapping": name_mapping,
            "extracted_names": extracted_names
        }
        
    except Exception as e:
        print(f"Name conversion failed: {str(e)}")
        return {"tamil_names": [], "name_mapping": {}, "extracted_names": []}

def get_conversation_context(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract context from previous messages"""
    if not messages or len(messages) < 2:
        return {"recent_exchanges": [], "topics_covered": [], "previous_queries": [], "conversation_flow": "new_topic"}
    
    # Get last 6 messages (3 exchanges) for context
    recent_messages = messages[-6:]
    
    context_parts = []
    queries_discussed = []
    topics_covered = []
    
    for i, msg in enumerate(recent_messages):
        if msg["role"] == "user":
            context_parts.append(f"User asked: {msg['content']}")
            
            # Extract topics from user questions
            content_lower = msg['content'].lower()
            if any(word in content_lower for word in ['male', 'ஆண்', 'men']):
                topics_covered.append('male_demographics')
            if any(word in content_lower for word in ['female', 'பெண்', 'women']):
                topics_covered.append('female_demographics')
            if any(word in content_lower for word in ['age', 'young', 'old', 'senior']):
                topics_covered.append('age_analysis')
            if any(word in content_lower for word in ['name', 'relation', 'tamil', 'named', 'called']):
                topics_covered.append('name_search')
            if any(word in content_lower for word in ['count', 'total', 'how many']):
                topics_covered.append('counting')
                
        elif msg["role"] == "assistant" and i > 0:
            # Try to extract SQL queries from assistant responses
            if "SELECT" in msg['content'].upper():
                sql_match = re.search(r'SELECT.*?FROM.*?(?=\n|$)', msg['content'], re.IGNORECASE | re.DOTALL)
                if sql_match:
                    queries_discussed.append(sql_match.group(0))
    
    context_summary = {
        "recent_exchanges": context_parts[-4:],  # Last 2 exchanges
        "topics_covered": list(set(topics_covered)),
        "previous_queries": queries_discussed[-2:],  # Last 2 SQL queries
        "conversation_flow": "follow_up" if len(context_parts) > 1 else "new_topic"
    }
    
    return context_summary

def analyze_question_intent(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if this is a follow-up question or new topic"""
    follow_up_indicators = [
        'more details', 'break down', 'drill down', 'show me', 'what about',
        'also', 'additionally', 'further', 'deeper', 'expand on', 'elaborate',
        'same but', 'similar', 'compare', 'vs', 'difference', 'contrast',
        'those', 'them', 'it', 'this', 'that', 'previous', 'above', 'earlier'
    ]
    
    question_lower = question.lower()
    is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)
    
    # Also check if question is very short (likely referring to previous context)
    if len(question.split()) <= 3 and not any(word in question_lower for word in ['total', 'count', 'show', 'list']):
        is_follow_up = True
    
    return {
        "is_follow_up": is_follow_up,
        "intent_type": "follow_up" if is_follow_up else "new_query",
        "context_relevance": "high" if context['conversation_flow'] == "follow_up" else "low"
    }

def generate_sql_query(question: str, tamil_names: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> str:
    """Generate SQL query with conversation context awareness and Tamil name support"""
    client = get_llm_client()
    schema_info = get_schema_info()
    schema_text = json.dumps(schema_info["schema"], indent=2)
    
    if context is None:
        context = {"recent_exchanges": [], "topics_covered": [], "previous_queries": [], "conversation_flow": "new_topic"}
    
    intent = analyze_question_intent(question, context)
    
    # Add Tamil names context if available
    tamil_names_context = ""
    if tamil_names:
        tamil_names_context = f"""
        
        ENGLISH-TO-TAMIL NAME CONVERSION:
        The user's question contained English names that have been converted to Tamil:
        Tamil names to search for: {', '.join(tamil_names)}
        
        IMPORTANT: Use these Tamil names in your SQL queries with LIKE or ILIKE operations for name matching.
        Example: WHERE name LIKE '%குமார்%' OR name LIKE '%ராஜ்%'
        """
    
    prompt = f"""
    Given this database schema for Tamil voter data:
    {schema_text}
    
    CONVERSATION CONTEXT:
    {json.dumps(context, indent=2)}
    
    CURRENT QUESTION: "{question}"
    {tamil_names_context}
    
    INTENT ANALYSIS:
    - Is this a follow-up question? {intent['is_follow_up']}
    - Intent type: {intent['intent_type']}
    - Context relevance: {intent['context_relevance']}
    
    CONTEXTUAL CHAIN OF THOUGHT:
    
    1. UNDERSTAND CONTEXT:
       - What topics have we discussed? {context.get('topics_covered', [])}
       - What was the last query about?
       - Is this building on previous analysis?
    
    2. INTERPRET CURRENT QUESTION:
       - If follow-up: How does this relate to previous queries?
       - If new topic: What new analysis is being requested?
       - What specific data is needed?
       - Are there converted Tamil names to use?
    
    3. IDENTIFY KEY ELEMENTS:
       - Gender values: 'ஆண்' (Male), 'பெண்' (Female)
       - Age groups: Young (18-36), Senior (>36), All adults (>=18)
       - Text searches: Names, relations in Tamil (including converted names)
       - Location: AC codes, street numbers, house numbers
    
    4. DETERMINE QUERY STRATEGY:
       - For follow-ups: Build upon or modify previous queries
       - For new topics: Start fresh analysis
       - For comparisons: Use GROUP BY or multiple conditions
       - For drill-downs: Add more specific WHERE clauses
       - For name searches: Use LIKE/ILIKE with Tamil names
    
    5. BUILD THE CONTEXTUAL QUERY:
       - Consider what the user already knows
       - Provide complementary or deeper insights
       - If they asked about males before, they might want female comparison
       - If they saw totals, they might want breakdowns
       - Use converted Tamil names for accurate matching
    
    IMPORTANT CONTEXT:
    - Table name: 'data'
    - Gender: 'ஆண்' (Male) and 'பெண்' (Female)
    - Columns: sno, street_no, name, relation_type, relation_name, house_no, age, gender, voter_id, voter_box, file_name, addition_page, ac_code
    - For name searches, use ILIKE for case-insensitive matching
    - Tamil names should be used with partial matching (LIKE '%name%')
    
    Generate the most contextually appropriate SQL query based on the conversation flow and any converted Tamil names.
    Return ONLY the SQL query.
    """
    
    response = client.invoke(prompt)
    return response.content.strip()

def execute_sql_query(sql_query: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as list of dictionaries"""
    try:
        conn = init_database()
        # Clean SQL query
        sql_query = re.sub(r'```sql\n?|```\n?', '', sql_query).strip()
        results_df = conn.execute(sql_query).df()
        return results_df.to_dict('records')
    except Exception as e:
        raise Exception(f"SQL execution failed: {str(e)}")

def explain_results(question: str, query: str, results: List[Dict[str, Any]], used_name_conversion: bool = False, context: Optional[Dict[str, Any]] = None) -> str:
    """Generate concise explanation that references conversation history and name conversion"""
    client = get_llm_client()
    results_summary = f"Query returned {len(results)} rows."
    
    if context is None:
        context = {"recent_exchanges": [], "topics_covered": [], "previous_queries": [], "conversation_flow": "new_topic"}
    
    intent = analyze_question_intent(question, context)
    
    conversion_note = ""
    if used_name_conversion:
        conversion_note = " (Note: English names were automatically converted to Tamil for accurate database matching)"
    
    prompt = f"""
    Question: "{question}"
    Results: {results_summary}
    Previous topics: {context.get('topics_covered', [])}
    Is follow-up: {intent['is_follow_up']}
    Used name conversion: {used_name_conversion}
    
    Provide a brief 2-3 sentence explanation of the results. If this is a follow-up question, briefly reference how it connects to our previous discussion. If name conversion was used, mention that English names were converted to Tamil for matching.{conversion_note} Keep it concise and conversational.
    """
    
    response = client.invoke(prompt)
    return response.content

def generate_follow_up_suggestions(question: str, query: str, results: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate smart follow-up question suggestions"""
    client = get_llm_client()
    
    if context is None:
        context = {"recent_exchanges": [], "topics_covered": [], "previous_queries": [], "conversation_flow": "new_topic"}
    
    prompt = f"""
    Based on our conversation about Tamil voter data:
    
    RECENT CONTEXT: {context.get('recent_exchanges', [])}
    LAST QUESTION: "{question}"
    LAST QUERY: {query}
    RESULTS COUNT: {len(results)} records
    
    Generate 3-4 smart follow-up questions that would naturally continue this conversation:
    
    Consider:
    - Drilling deeper into the current results
    - Comparing with different demographics
    - Exploring related aspects (age, location, names)
    - Building on patterns we've discovered
    - Name-based searches (both English and Tamil)
    
    Format as a simple list of questions that I can display as clickable buttons.
    Each question should be concise (under 50 characters) and build logically on our analysis.
    """
    
    response = client.invoke(prompt)
    
    # Extract clean questions from the response
    questions = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('Based on'):
            # Clean up numbering and formatting
            clean_line = re.sub(r'^\d+\.?\s*', '', line)
            clean_line = re.sub(r'^-\s*', '', clean_line)
            clean_line = clean_line.strip('"\'')
            if clean_line and len(clean_line) > 10:
                questions.append(clean_line)
    
    return questions[:4]  # Return max 4 suggestions