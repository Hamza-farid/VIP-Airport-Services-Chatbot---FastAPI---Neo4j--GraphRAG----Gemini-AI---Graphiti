"""
User Question ->
Gemini (extract entities) -> 
If entities found: Neo4j (GraphRAG) -> Use results
If no entities/no results: Full-text search in Neo4j
Final Answer -> User
"""
import httpx
import os
import sys
import json
import time
import uuid
import hashlib
import logging
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from neo4j import GraphDatabase

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

# ========== Graphiti Configuration ==========
def configure_graphiti():
    api_key = os.getenv("GOOGLE_API_KEY")  # Changed from GEMINI_API_KEY
    
    return Graphiti(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        llm_client=GeminiClient(
            config=LLMConfig(
                api_key=api_key,
                model="gemini-2.0-flash"
            )
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key,
                embedding_model="embedding-001"
            )
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(
                api_key=api_key,
                model="gemini-2.0-flash-exp"
            )
        )
    )

# Initialize Graphiti
graphiti = configure_graphiti()

# Configure Gemini API directly (for non-Graphiti uses)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]
model = genai.GenerativeModel('gemini-2.0-flash',safety_settings=safety_settings)


HASH_FILE = "file_hash.txt"

SYSTEM_PROMPT = """
You are an expert assistant specialized in airport VIP services. Answer user questions in 1 to 4 sentences, maximum 4 lines. Be clear, informative, and avoid repetition. Output should be concise and complete as a single paragraph.
1. For greetings (hi, hello, etc):
   - Respond warmly but briefly (1 sentence)
   - Example: "Hello! How can I assist you with airport services today?"
2. For service questions:
   - Answer in 1-4 concise sentences
   - Be specific about VIP services available
   - Example: "We offer VIP meet-and-greet services at Dubai Airport including fast track security and lounge access." 
3. For unclear requests:
   - Politely ask for clarification
   - Example: "Could you specify which airport service you're interested in?" 
4. Never respond with empty content

- If the user is asking about availability in a city, store it as an interest (not their current city). However, if they ask about flight availability *from* a city, you may infer it as their current location.
"""

# ========== Helper Functions ==========
def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def has_file_changed(filepath):
    current_hash = get_file_hash(filepath)
    try:
        with open(HASH_FILE, "r") as f:
            previous_hash = f.read().strip()
    except FileNotFoundError:
        previous_hash = None
    if current_hash != previous_hash:
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
        return True
    return False

def convert_to_gemini_messages(openai_style_messages):
    gemini_messages = []
    for msg in openai_style_messages:
        prefix = "system: " if msg.get("role") == "system" else ""
        gemini_messages.append({
            "role": "user",
            "parts": [f"{prefix}{msg.get('content')}"]
        })
    return gemini_messages

async def reset_neo4j_schema():
    try:
        await graphiti.build_indices_and_constraints()
        logger.info("🔷 Neo4j constraints & indexes applied")
    except Exception as e:
        logger.warning(f"🔶 Neo4j schema warning: {e}")

async def generate_response_with_retry(user_input, context=None, chat_history=None, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return await generate_response(user_input, context, chat_history)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"⚠️ Gemini rate limit reached. Retry {attempt+1}/{max_retries} in {delay}s")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise
    logger.error("❌ Failed to generate response after retries")
    return {"answer": "Sorry, I'm having trouble responding right now. Please try again later."}

async def generate_response(user_input, context=None, chat_history=None):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": "assistant" if msg["role"] == "CHATBOT" else "user",
                    "content": msg["message"]
                })
        if context:
            user_input = f"Context:\n{context}\n\nQuestion: {user_input}"
        messages.append({"role": "user", "content": user_input})
        gemini_messages = convert_to_gemini_messages(messages)
        
        response = model.generate_content(gemini_messages)
        response_text = getattr(response, 'text', '').strip()
        
        if not response_text:
            return {"answer": "Sorry, I couldn't generate a response."}
        
        try:
            if response_text.startswith('{') and response_text.endswith('}'):
                parsed = json.loads(response_text)
                if 'answer' in parsed:
                    return parsed
                return {"answer": response_text}
            return {"answer": response_text}
        except json.JSONDecodeError:
            return {"answer": response_text}
    except Exception as e:
        logger.exception("generate_response failed")
        return {"answer": f"Sorry, something went wrong: {str(e)}"}

def get_weather_by_city(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"answer": "Weather service is not configured."}
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather = data["weather"][0]["description"]
                temp_c = data["main"]["temp"]
                temp_f = temp_c * 9/5 + 32
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                message = (
                    f"The weather in {city} is currently '{weather}'. "
                    f"Temperature: {temp_c:.1f}°C ({temp_f:.1f}°F), "
                    f"Humidity: {humidity}%, Wind: {wind_speed} m/s."
                )
                return {"answer": message}
            elif response.status_code == 429:
                logger.warning(f"⚠️ Weather API rate limit reached. Attempt {attempt+1}/3")
                time.sleep(2 ** attempt)
            else:
                return {"answer": f"Weather data for '{city}' is unavailable."}
        except Exception as e:
            logger.error(f"❌ Error calling weather API: {str(e)}")
            time.sleep(1)
    
    return {"answer": "Weather service error occurred. Please try again later."}

async def load_and_index_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    sections = [s.strip() for s in raw_text.split('---') if s.strip()]
    
    for i, section in enumerate(sections):
        try:
            await graphiti.add_episode(
                name=f"doc_section_{i}",
                episode_body=section,
                source=EpisodeType.text,
                source_description="VIP Service Document Section",
                reference_time=datetime.now(timezone.utc))
            if (i + 1) % 5 == 0:
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"❌ Failed to add document section: {str(e)}")
    logger.info(f"✅ Processed {len(sections)} document sections into Neo4j")

# ========== FastAPI Lifespan ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🛠️ Initializing application...")
    
    try:
        await reset_neo4j_schema()
        logger.info("🔷 Neo4j: Schema verified")
    except Exception as e:
        logger.warning(f"🔶 Neo4j: Schema notes - {str(e)}")

    try:
        if os.path.exists("file.txt") and has_file_changed("file.txt"):
            logger.info("📄 Processing document updates...")
            await load_and_index_documents("file.txt")
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")

    logger.info("🚀 All systems ready")
    yield
    logger.info("🛑 Application shutting down")

# ========== FastAPI App ==========
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {"status": "active", "system": "VIP Airport Services Chatbot"}
@app.post("/messages")
async def handle_message(req: MessageRequest):
    raw_input = req.message.strip()
    raw_input = raw_input.lower()
    logger.info(f"📥 Received user message: {raw_input}")

    session_id = "test-session-123"  # In production, generate or get from request

# =============================== Entity Extraction =====================================================
    extraction_prompt = """
Extract the following fields ONLY if present in the user input:
- current_location: If the user mentions where they are flying from
- interests: If asking about a city/airport/service
- airport: Airport name or IATA code
- flight_number: Flight number if provided
- date: Any date or flight date
- service_requested: Service type (VIP meet and greet, etc)
- pickup/dropoff_location: If mentioned
- time: Time of pickup/flight
- passenger_name: If mentioned

Return as flat JSON. Only include present fields.
"""
    extracted = {}
    try:
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": raw_input}
        ]
        response = model.generate_content(convert_to_gemini_messages(messages))
        raw_text = getattr(response, 'text', '').strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        try:
            extracted = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            try:
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    extracted = json.loads(raw_text[start_idx:end_idx+1])
            except:
                extracted = {}
#-------------------- Store in Neo4j----------------------
        await graphiti.add_episode(
            name=f"user_input_{session_id}_{uuid.uuid4()}",
            episode_body=raw_input,
            source=EpisodeType.text,
            source_description="User message",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f"✅ Entities extracted: {list(extracted.keys())}" if extracted else "⚠️ No entities extracted")

    except Exception as e:
        logger.exception("🚨 Error during extraction")
        extracted = {}
#---------------------------------------Routing---------------------------------------
# =============================NEW: LLM Task Routing ==================================
    routing_prompt = f"""
Decide the correct task for the user's request.
Options:
- weather
- general_qa

Criteria:
- If the user asks about current weather, temperature, forecast, or climate in a location -> weather
- Otherwise -> general_qa

Respond ONLY with one word: weather or general_qa.

User message: "{raw_input}"
"""
    try:
        route_resp = model.generate_content(convert_to_gemini_messages([{"role": "user", "content": routing_prompt}]))
        route_decision = getattr(route_resp, 'text', '').strip().lower()
        logger.info(f"🔀 LLM route decision: {route_decision}")
    except Exception as e:
        logger.error(f"❌ Routing decision failed: {e}")
        route_decision = "general_qa"

    # ========== WEATHER PATH ==========
    if route_decision == "weather":
        city = extracted.get("current_location") or extracted.get("interests")
        if not city:
            return JSONResponse(content={"answer": "Could you tell me which city you're asking about?"})
        
        logger.info(f"🌦️ Fetching weather for city: {city}")
        weather_data = get_weather_by_city(city)
        
        # Feed weather info to LLM for nice response
        weather_prompt = f"""
-You are an helpful assistant. The user asked about the weather.
-Weather data:
-{weather_data['answer']}

-transform that data in a better way for better understanding of the user.
-keep tone helpful and polite. 
-also ask what else you can do for the user.
"""
        final_weather_resp = model.generate_content(convert_to_gemini_messages([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": weather_prompt}
        ]))
        final_text = getattr(final_weather_resp, 'text', '').strip()
        return JSONResponse(content={"answer": final_text})

    # ========== GENERAL_QA PATH ==========
    context = None
    try:
        if extracted:
            entity_filters = " ".join(f"{k}:{v}" for k,v in extracted.items() if v)
            combined_query = f"{raw_input} {entity_filters}"
            logger.info("🔍 Searching Neo4j with extracted entities...")
            results = await graphiti.search(query=combined_query)
        else:
            logger.info("🔍 Performing full-text search in Neo4j...")
            results = await graphiti.search(query=raw_input)

        facts = []
        for edge in results[:10]:  # Limit to top 10 results
            fact = getattr(edge, "fact", None) or getattr(edge, "text", None) or str(edge)
            if fact and len(fact) > 20:
                facts.append(fact[:300])

        if facts:
            context = "\n\n".join(facts[:5])
            logger.info(f"🔍 Found {len(facts)} relevant facts")

    except Exception as e:
        logger.error(f"Search failed: {e}")

    response_data = await generate_response_with_retry(
        user_input=raw_input,
        context=context
    )
    
    return JSONResponse(content=response_data)
