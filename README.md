# ✈️ VIP Airport Services Chatbot  
_A FastAPI-based intelligent assistant for airport VIP services using Neo4j (GraphRAG), Gemini AI, and Graphiti._

## 🚀 Overview
This project implements an AI-powered chatbot designed to assist users with **VIP airport services** such as meet-and-greet, lounge access, concierge logistics, and weather updates.  

The system integrates:  
- **FastAPI** → Backend framework for handling API requests  
- **Neo4j (GraphRAG via Graphiti)** → Knowledge graph storage & retrieval  
- **Gemini AI (LLM + embeddings)** → Entity extraction, response generation, reranking  
- **OpenWeather API** → Provides real-time weather updates for airports/cities  
- **Graphiti** → Manages episodes, embeddings, and schema constraints in Neo4j  

The chatbot can:  
- Answer service-related questions concisely  
- Extract structured entities (airport, city, date, service type, etc.)  
- Route queries between **general Q&A** and **weather lookups**  
- Store and search knowledge in **Neo4j GraphRAG**  

---

## ⚙️ Features
✅ Airport VIP services Q&A (meet & greet, transfers, lounges, etc.)  
✅ Entity extraction with Gemini AI (current location, airport, service type, etc.)  
✅ Intelligent query routing (Weather vs. General Q&A)  
✅ Weather integration (OpenWeather API)  
✅ Knowledge graph storage in **Neo4j** via **Graphiti**.  
✅ Retry logic for LLM rate limits.  
✅ Document ingestion (`file.txt`) with automatic hashing & re-indexing.  

---

## 🛠️ Tech Stack
- **Backend**: FastAPI  
- **Database**: Neo4j (GraphRAG with Graphiti)  
- **LLM**: Google Gemini (via `google.generativeai`)  
- **Embeddings & Reranking**: Gemini Embedder + Gemini Reranker  
- **Weather API**: OpenWeather  
- **Infra**: Async Python, logging, CORS support  

---
