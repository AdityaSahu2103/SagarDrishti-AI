"""
RAG (Retrieval-Augmented Generation) engine for ARGO oceanographic data queries.
Combines vector search with an OpenAI LLM for intelligent data exploration.
Handles both general conversational queries and data-specific questions.
"""

import os
from typing import List, Dict, Optional

# Import LangChain components with comprehensive error handling
ChatOpenAI = None
HumanMessage = None
SystemMessage = None

# Try multiple import strategies
try:
    # First try: Modern langchain-openai
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    print("✅ Successfully imported ChatOpenAI and message classes")
except ImportError:
    try:
        # Second try: Legacy langchain structure
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        print("✅ Imported ChatOpenAI from legacy langchain structure")
    except ImportError:
        try:
            # Third try: Community version
            from langchain_community.chat_models import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            print("✅ Imported ChatOpenAI from langchain_community")
        except ImportError:
            try:
                # Fourth try: Direct OpenAI
                import openai
                print("⚠️ Using direct OpenAI client as fallback")
                
                class ChatOpenAI:
                    def __init__(self, model="gpt-3.5-turbo-instruct", temperature=0.2, max_tokens=512, api_key=None):
                        self.client = openai.OpenAI(api_key=api_key)
                        self.model = model
                        self.temperature = temperature
                        self.max_tokens = max_tokens
                    
                    def invoke(self, messages):
                        # Convert messages to OpenAI format
                        openai_messages = []
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                role = "system" if isinstance(msg, SystemMessage) else "user"
                                openai_messages.append({"role": role, "content": msg.content})
                        
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=openai_messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        
                        class Response:
                            def __init__(self, content):
                                self.content = content
                        
                        return Response(response.choices[0].message.content)
                
                class HumanMessage:
                    def __init__(self, content):
                        self.content = content
                
                class SystemMessage:
                    def __init__(self, content):
                        self.content = content
                        
            except ImportError:
                print("⚠️ Could not import any LLM libraries")
                print("⚠️ LLM (Language Model) didnt initialize. The chat interface will use a simplified, rule-based response mode.")
                
                # Create simple fallback message classes
                class HumanMessage:
                    def __init__(self, content):
                        self.content = content
                
                class SystemMessage:
                    def __init__(self, content):
                        self.content = content

from config import Config
try:
    from mcp_client import MCPClient
except Exception:
    MCPClient = None  # type: ignore
from embedding_index import ProfileEmbeddingIndex
from utils import setup_logging, timing_decorator

# Handle QueryAnalyzer import
try:
    from utils import QueryAnalyzer
except ImportError:
    QueryAnalyzer = None
    print("⚠️ QueryAnalyzer not available")

logger = setup_logging(__name__)

class OceanographyRAG:
    def __init__(self, embedding_index: ProfileEmbeddingIndex, config: Config = Config, mcp_client: Optional["MCPClient"] = None):
        self.embedding_index = embedding_index
        self.config = config
        self.llm = None
        self.openai_available = ChatOpenAI is not None
        self.mcp_client = mcp_client

    @staticmethod
    def _region_from_coords(latitude: float, longitude: float) -> str:
        """Classify ocean basin globally based on coordinates.
        Returns labels like 'North Pacific Ocean', 'South Atlantic Ocean', 'Indian Ocean',
        'Southern Ocean', or 'Arctic Ocean'. Mirrors dashboard.get_region().
        """
        try:
            lat = float(latitude)
            lon = float(longitude)
        except Exception:
            return "Unknown"

        # Normalize longitude to [-180, 180]
        if lon > 180:
            lon = ((lon + 180) % 360) - 180
        if lon < -180:
            lon = ((lon - 180) % 360) + 180

        # Polar oceans
        if lat >= 66.0:
            return "Arctic Ocean"
        if lat <= -50.0:
            return "Southern Ocean"

        # Main basins by longitude
        if -70.0 <= lon <= 20.0:
            basin = "Atlantic Ocean"
        elif 20.0 < lon < 146.0:
            basin = "Indian Ocean"
        else:
            basin = "Pacific Ocean"

        if basin in ("Atlantic Ocean", "Pacific Ocean"):
            hemi = "North" if lat >= 0 else "South"
            return f"{hemi} {basin}"
        return basin

    @staticmethod
    def _format_month_year(dt) -> str:
        """Format a datetime-like object as MON YYYY in uppercase, fallback to ISO date if unavailable."""
        try:
            return dt.strftime('%b %Y').upper()
        except Exception:
            try:
                return str(getattr(dt, 'date', lambda: dt)())
            except Exception:
                return str(dt)

    def _available_oceans(self) -> list[str]:
        """Return a sorted list of high-level oceans present in the dataset based on profile coordinates."""
        oceans: list[str] = []
        try:
            metas = self.embedding_index.profile_metadata or []
            for m in metas:
                lat = m.get('latitude'); lon = m.get('longitude')
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    region = self._region_from_coords(lat, lon) or ""
                    # Normalize to high-level oceans
                    if "Atlantic" in region:
                        oceans.append("Atlantic Ocean")
                    elif "Pacific" in region:
                        oceans.append("Pacific Ocean")
                    elif "Indian" in region:
                        oceans.append("Indian Ocean")
                    elif "Arctic" in region:
                        oceans.append("Arctic Ocean")
                    elif "Southern" in region:
                        oceans.append("Southern Ocean")
        except Exception:
            pass
        return sorted(set([o for o in oceans if o]))

    def _detect_requested_ocean(self, query: str) -> str | None:
        """Detect if the query explicitly mentions a high-level ocean. Returns standardized name or None."""
        q = (query or "").lower()
        if any(k in q for k in ["pacific", "north pacific", "south pacific"]):
            return "Pacific Ocean"
        if any(k in q for k in ["atlantic", "north atlantic", "south atlantic"]):
            return "Atlantic Ocean"
        if "indian" in q:
            return "Indian Ocean"
        if "arctic" in q:
            return "Arctic Ocean"
        if "southern ocean" in q or "antarctic" in q:
            return "Southern Ocean"
        return None

    def _get_profiles_by_ocean(self, ocean: str, limit: int = 3) -> List[Dict]:
        """Return up to 'limit' profiles formatted like retrieval results for a given high-level ocean.
        Matching is based on computed region from lat/lon in metadata.
        """
        results: List[Dict] = []
        if not ocean:
            return results
        try:
            metas = self.embedding_index.profile_metadata or []
            sums = self.embedding_index.profile_summaries or []
            count = 0
            for i, meta in enumerate(metas):
                try:
                    lat = meta.get('latitude'); lon = meta.get('longitude')
                    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                        continue
                    region = self._region_from_coords(lat, lon) or ""
                    # Accept either exact ocean label or hemisphere+ocean (e.g., 'North Pacific Ocean')
                    if ocean in region or ocean.split()[0] in region:
                        results.append({
                            'similarity_score': 1.0,
                            'distance': 0.0,
                            'summary': sums[i] if i < len(sums) else "",
                            'metadata': meta
                        })
                        count += 1
                        if count >= max(1, limit):
                            break
                except Exception:
                    continue
        except Exception:
            return []
        return results

    def _apply_mcp_filters(self, profiles: List[Dict], nl_query: str) -> tuple[List[Dict], Optional[Dict]]:
        """Use MCP to generate structured filters and apply them to retrieved profiles.
        Returns (filtered_profiles, mcp_plan) where mcp_plan contains SQL and filters.
        """
        if not self.mcp_client or not getattr(self.config, "MCP_ENABLED", False):
            return profiles, None
        try:
            plan = self.mcp_client.tool_argo_sql_query(nl_query)
            filters = plan.get("filters", [])
            # Always attach our DB-specific SQL for transparency/copy-paste
            try:
                plan["sql"] = self._filters_to_sql(filters)
            except Exception:
                pass
            if not filters:
                return profiles, plan

            def match(meta: Dict) -> bool:
                # meta may include latitude, longitude, time
                region = None
                lat = meta.get("latitude")
                lon = meta.get("longitude")
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    region = self._region_from_coords(lat, lon)
                t = meta.get("time")
                mo = getattr(t, "month", None)
                yr = getattr(t, "year", None)

                for f in filters:
                    fld, op, val = f.get("field"), f.get("op"), f.get("value")
                    if fld == "region" and region is not None:
                        if op == "=" and region != val:
                            return False
                    if fld == "month" and mo is not None:
                        if op == "=" and mo != val:
                            return False
                    if fld == "year" and yr is not None:
                        if op == "=" and yr != val:
                            return False
                    # has_temperature/has_salinity are always true for our dataset if present in metadata
                return True

            filtered = []
            for p in profiles:
                meta = p.get("metadata", {})
                if match(meta):
                    filtered.append(p)
            # If filters remove everything, keep original to avoid empty UX
            return (filtered or profiles), plan
        except Exception:
            return profiles, None

    def _filters_to_sql(self, filters: List[Dict]) -> str:
        """Translate MCP filters into a SQL query for our PostgreSQL schema.
        Maps common fields:
          - month, year -> EXTRACT from argo_profiles.time
          - latitude, longitude comparisons -> argo_profiles.latitude/longitude
          - has_salinity -> EXISTS salinity in argo_measurements
          - has_temperature -> EXISTS temperature in argo_measurements
        Returns a copy‑pastable SQL string.
        """
        base = [
            "SELECT p.id, p.file_source, p.profile_idx, p.latitude, p.longitude, p.time,",
            "       p.n_measurements, p.depth_range, p.temp_range, p.summary",
            "FROM public.argo_profiles p",
        ]
        where: List[str] = []

        for f in filters or []:
            fld = str(f.get("field", "")).lower().strip()
            op = str(f.get("op", "=")).strip()
            val = f.get("value")

            # Normalize operator symbols
            if op in {"eq"}: op = "="
            if op in {"gte"}: op = ">="
            if op in {"lte"}: op = "<="

            if fld == "month" and val is not None:
                where.append(f"EXTRACT(MONTH FROM p.time) = {int(val)}")
            elif fld == "year" and val is not None:
                where.append(f"EXTRACT(YEAR FROM p.time) = {int(val)}")
            elif fld == "latitude" and val is not None:
                # op with numeric
                where.append(f"p.latitude {op} {float(val)}")
            elif fld == "longitude" and val is not None:
                where.append(f"p.longitude {op} {float(val)}")
            elif fld == "has_salinity":
                cond = (
                    "EXISTS (SELECT 1 FROM public.argo_measurements m"
                    " WHERE m.profile_id = p.id AND m.salinity IS NOT NULL)"
                )
                where.append(cond if bool(val) else f"NOT ({cond})")
            elif fld == "has_temperature":
                cond = (
                    "EXISTS (SELECT 1 FROM public.argo_measurements m"
                    " WHERE m.profile_id = p.id AND m.temperature IS NOT NULL)"
                )
                where.append(cond if bool(val) else f"NOT ({cond})")
            # Unknown fields are ignored for SQL but may still be applied client-side

        sql = "\n".join(base)
        if where:
            sql += "\nWHERE " + " AND ".join(where)
        sql += "\nORDER BY p.time ASC;"
        return sql

    def initialize_llm(self):
        """Initialize the OpenAI Chat LLM with comprehensive fallback handling."""
        if not self.openai_available:
            logger.warning("OpenAI ChatGPT not available - using rule-based responses")
            print("⚠️ LLM (Language Model) didnt initialize. The chat interface will use a simplified, rule-based response mode.")
            return
            
        if self.llm is None:
            if not self.config.OPENAI_API_KEY:
                logger.error("OpenAI API key not found in configuration!")
                print("⚠️ LLM (Language Model) didnt initialize. The chat interface will use a simplified, rule-based response mode.")
                self.openai_available = False
                return

            logger.info(f"Initializing OpenAI LLM: {self.config.LLM_MODEL}")

            try:
                # Initialize with the available ChatOpenAI class (could be LangChain or direct OpenAI)
                module_name = getattr(ChatOpenAI, "__module__", "")
                logger.info(f"Using ChatOpenAI from module: {module_name}")

                init_kwargs = {
                    "model": self.config.LLM_MODEL,
                    "temperature": 0.2,
                    "max_tokens": self.config.LLM_MAX_TOKENS,
                }

                # Try modern LangChain signature first
                try:
                    self.llm = ChatOpenAI(**init_kwargs, openai_api_key=self.config.OPENAI_API_KEY)
                    logger.info("Initialized ChatOpenAI with openai_api_key parameter.")
                except TypeError:
                    # Try fallback signature (our direct OpenAI shim or other variants)
                    try:
                        self.llm = ChatOpenAI(**init_kwargs, api_key=self.config.OPENAI_API_KEY)
                        logger.info("Initialized ChatOpenAI with api_key parameter.")
                    except TypeError:
                        # As a last resort, rely on environment variable
                        self.llm = ChatOpenAI(**init_kwargs)
                        logger.info("Initialized ChatOpenAI using environment variable for API key.")
                
                # Test the LLM with a simple query
                test_message = [HumanMessage(content="Hello")]
                test_response = self.llm.invoke(test_message)
                
                logger.info("✅ OpenAI LLM initialized and tested successfully.")
                print("✅ LLM (Language Model) initialized successfully!")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI LLM: {e}")
                print(f"⚠️ LLM initialization failed: {e}")
                print("⚠️ LLM (Language Model) didnt initialize. The chat interface will use a simplified, rule-based response mode.")
                self.llm = None
                self.openai_available = False

    def _is_general_conversation(self, query: str) -> bool:
        """Detect if the query is general conversation rather than a data query."""
        query_lower = query.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good day']
        system_questions = ['what can you do', 'help', 'how does this work', 'who are you']

        return any(greet in query_lower for greet in greetings) or \
               any(question in query_lower for question in system_questions)

    def _is_inventory_query(self, query: str) -> bool:
        """Detect queries asking about what data/profiles are available."""
        q = query.lower()
        keywords = [
            'what profiles', 'what all argo profiles', 'what argo profiles', 'what floats',
            'what data do you have', 'list profiles', 'show profiles', 'available profiles',
            'which profiles', 'dataset overview', 'catalog', 'how many profiles', 'what data you have'
        ]
        return any(k in q for k in keywords)

    def _match_general_info_intent(self, query: str) -> str | None:
        """Identify general info intents and return an intent key, else None."""
        q = query.lower()
        # Map of intents to keyword triggers
        intent_map = {
            'what_is_argo': ['what is argo', 'argo program', 'about argo', 'argo floats'],
            'how_to_use': ['how to use', 'how do i use', 'guide', 'tutorial', 'help me use'],
            'features': ['features', 'what can you do', 'capabilities', 'what does this app do'],
            'data_sources': ['data source', 'where data from', 'which data do you use', 'argo data source'],
            'regions': ['which regions', 'what regions', 'supported regions', 'areas covered'],
            'visualizations': ['what charts', 'what plots', 'visualizations', 'graphs available'],
            'privacy_limits': ['limits', 'limitations', 'accuracy', 'privacy', 'data privacy'],
            'commands_examples': ['examples', 'sample queries', 'what can i ask', 'give examples'],
        }
        for intent, keys in intent_map.items():
            if any(k in q for k in keys):
                return intent
        return None

    def _generate_general_info_answer(self, intent: str) -> str:
        """Return canned, concise, user-friendly answers for general info intents."""
        if intent == 'what_is_argo':
            return (
                "ARGO floats are small robotic instruments drifting in the ocean. They dive from the surface to deep water and measure temperature, salinity, and pressure."
                " This helps scientists understand ocean health, weather, and climate."
            )
        if intent == 'how_to_use':
            return (
                "Type a question in plain English (e.g., ‘Show temperature profiles in the Arabian Sea’)."
                " I’ll retrieve matching ARGO profiles, summarize them, show plots, and provide industry insights."
            )
        if intent == 'features':
            return (
                "Key features: AI chat for ocean questions, semantic search over ARGO profiles, interactive maps and plots, and simple industry insights based on the retrieved data."
            )
        if intent == 'data_sources':
            return (
                "Data comes from ARGO program NetCDF files you’ve loaded into the app (e.g., Indian Ocean profiles)."
            )
        if intent == 'regions':
            return (
                "Common regions include Arabian Sea, Bay of Bengal, Equatorial and Southern Indian Ocean. I infer regions from the profile coordinates."
            )
        if intent == 'visualizations':
            return (
                "I can show temperature/salinity vs depth, temperature–salinity (T–S) plots, maps of float locations, distributions by region, and simple time trends."
            )
        if intent == 'privacy_limits':
            return (
                "Results depend on the local ARGO files you loaded. I don’t share your data externally."
                " As with any model, insights are approximate and should be validated for critical decisions."
            )
        if intent == 'commands_examples':
            return (
                "Try: ‘Show temperature profiles in the Arabian Sea’, ‘Find floats with high salinity’, ‘Generate a T–S plot’, or ‘What profiles do you have?’."
            )
        return "Ask me about ocean temperature, salinity, regions, or what profiles are available."

    def _get_inventory_profiles(self, top_k: int) -> List[Dict]:
        """Return up to top_k representative profiles from the index without semantic search."""
        metas = self.embedding_index.profile_metadata
        summaries = self.embedding_index.profile_summaries
        results: List[Dict] = []
        if not metas or not summaries:
            return results
        count = min(top_k, len(metas))
        for i in range(count):
            results.append({
                'similarity_score': 1.0,  # neutral/default
                'distance': 0.0,
                'summary': summaries[i],
                'metadata': metas[i]
            })
        return results

    def _generate_inventory_answer(self, profiles: List[Dict]) -> str:
        """Generate a simple, human-friendly summary of available profiles."""
        if not profiles:
            return "No ARGO profiles are currently loaded. Please add data files and refresh."

        total = len(self.embedding_index.profile_metadata) if self.embedding_index.profile_metadata else 0
        # Compute dataset-wide context
        metas = self.embedding_index.profile_metadata or []
        regions: list[str] = []
        lats: list[float] = []
        lons: list[float] = []
        times = []
        file_samples: list[str] = []
        for m in metas:
            try:
                lat = m.get('latitude'); lon = m.get('longitude')
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    regions.append(self._region_from_coords(lat, lon))
                    lats.append(lat); lons.append(lon)
                fs = m.get('file_source')
                if fs and len(file_samples) < 5 and fs not in file_samples:
                    file_samples.append(str(fs))
                if m.get('time'):
                    times.append(m['time'])
            except Exception:
                pass
        regions = sorted(set([r for r in regions if r]))
        bbox = None
        if lats and lons:
            try:
                bbox = f"lat {min(lats):.2f}°–{max(lats):.2f}°, lon {min(lons):.2f}°–{max(lons):.2f}°"
            except Exception:
                bbox = None
        time_span = None
        if times:
            try:
                tmin, tmax = min(times), max(times)
                time_span = f"{tmin.date().isoformat()} to {tmax.date().isoformat()}"
            except Exception:
                time_span = None

        # Oceans summary (based on regions present)
        oceans = []
        if regions:
            # The current dataset uses Indian Ocean subregions; keep generic in case of future data
            if any(r for r in regions if 'Indian Ocean' in r or 'Arabian Sea' in r or 'Bay of Bengal' in r):
                oceans.append('Indian Ocean')

        parts = [
            "Summary: Here’s what’s available in the local ARGO dataset. 🌊",
            f"- 🔢 Total profiles: {total}",
            f"- 🗺️ Regions: {', '.join(regions) if regions else 'Unknown'}",
            *( [f"- 🌐 Oceans: {', '.join(sorted(set(oceans)))}"] if oceans else [] ),
            *( [f"- 📍 Coverage (bbox): {bbox}"] if bbox else [] ),
            *( [f"- 🕒 Time span: {time_span}"] if time_span else [] ),
            *( [f"- 📁 Sample floats/files: {', '.join(file_samples)}"] if file_samples else [] ),
            "",
            "Sample profiles (ID/location):"
        ]
        # Add a short list of IDs/locations
        examples = []
        for p in profiles[: min(5, len(profiles))]:
            meta = p.get('metadata', {})
            fid = meta.get('file_source', 'N/A')
            lat = meta.get('latitude')
            lon = meta.get('longitude')
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                examples.append(f"- {fid} at ({lat:.2f}°, {lon:.2f}°)")
            else:
                examples.append(f"- {fid}")
        if examples:
            parts.append("\n".join(examples))

        return "\n".join(parts)

    def _generate_conversational_response(self, question: str) -> str:
        """Generate a conversational response using LLM or fallback."""
        if not self.llm:
            self.initialize_llm()
            
        if not self.llm:
            # Fallback to rule-based response
            return self._generate_rule_based_greeting(question)

        system_prompt = (
            "You are FloatChat, an AI-powered multilingual assistant for exploring ARGO oceanographic data and answering general questions. "
            "You combine retrieval-augmented generation (RAG) with friendly explanations and clear recommendations.\n\n"
            "LANGUAGE SUPPORT:\n"
            "- Respond in the same language as the user's question (supports English, Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Gujarati, Punjabi).\n"
            "- Handle Hinglish and code-mixed Indian languages smoothly; understand transliterated terms (e.g., 'taapman' for temperature, 'namak/kharaapan' for salinity).\n"
            "- If the question mixes languages, default to English but include important technical terms in the local language when helpful.\n\n"
            "OBJECTIVES:\n"
            "1. Grounding: Prefer retrieved ARGO float profile summaries. Do not hallucinate values.\n"
            "2. Clarity: Always provide a one-line summary first, then structured details.\n"
            "3. Multilingual Support: Switch languages seamlessly; keep technical terms in English if no common translation exists.\n"
            "4. General-Purpose Support: If no relevant context is found, gracefully answer using general knowledge and explicitly mention that no local ARGO data was found.\n"
            "5. Next-Step Guidance: Suggest follow-up queries or visualizations (e.g., 'plot temperature vs depth' or 'map float locations').\n"
            "6. Transparency: If confidence is low, say so. If data is missing, suggest filters (region, time, parameter).\n\n"
            "LENGTH & STYLE:\n"
            "- Aim for around 7–8 sentences total across sections.\n"
            "- Include 2–4 relevant emojis to improve readability (e.g., 🌊, 📊, 🗺️, 🧭).\n\n"
            "RESPONSE STRUCTURE:\n"
            "- Summary: One clear line answering the query.\n"
            "- Details: 2–4 short bullet points citing relevant numbers, regions, depths, times.\n"
            "- Always include the named oceanic region when possible (e.g., Arabian Sea, Bay of Bengal, Equatorial Indian Ocean).\n"
            "- Format months as three-letter uppercase (JAN, FEB, …) with year (e.g., JAN 2023).\n"
            "- Next Step: Suggest how to refine or explore further.\n\n"
            "RULES OF CONDUCT:\n"
            "- Never make up numbers — rely on retrieved data.\n"
            "- Stay polite, concise, beginner-friendly.\n"
            "- If data is too technical, explain simply.\n"
            "- Be robust for off-topic questions (answer intelligently).\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "USER: 'Show me salinity profiles near the equator in March 2023.'\n"
            "ASSISTANT:\n"
            "Summary: Salinity near the equator in March 2023 was stable and mid-range.\n"
            "- Profiles sampled between 0°–5° latitude, depth 0–2000 m.\n"
            "- Salinity range: 34.5–35.0 PSU across most profiles.\n"
            "- Concentration highest near 20°E–30°E longitude.\n\n"
            "\n"
            "USER: 'अरब सागर में पिछले महीने का तापमान क्या था?'\n"
            "ASSISTANT:\n"
            "Summary: अरब सागर का औसत तापमान पिछले महीने मध्यम था।\n"
            "- सतह तापमान ~28°C–29°C।\n"
            "- गहराई के साथ तापमान धीरे-धीरे घटता है।\n"
            "- अधिकांश प्रोफाइल 0–2000 m गहराई तक।\n\n"
            "\n"
            "USER: 'Who is the Prime Minister of India?'\n"
            "ASSISTANT: Summary: Currently, the Prime Minister of India is Shri Narendra Modi (as of 2025).\n"
            "(This answer is general knowledge, not based on ARGO data.)\n"
        )

        user_prompt = (
            f"Conversation opener. The user said: '{question}'.\n"
            "- Introduce yourself briefly as FloatChat.\n"
            "- Follow the RESPONSE STRUCTURE.\n"
            "- If the user seems off-topic, answer intelligently using general knowledge and state that this is not from local ARGO data.\n"
            "- Keep it concise and friendly."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            # Try different invocation methods
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(messages)
            elif hasattr(self.llm, '__call__'):
                response = self.llm(messages)
            else:
                # Fallback for very old versions
                response = self.llm.predict([msg.content for msg in messages])
                return response
                
            return response.content.strip()
        except Exception as e:
            logger.error(f"Conversational LLM generation failed: {e}")
            return self._generate_rule_based_greeting(question)

    def _generate_rule_based_greeting(self, question: str) -> str:
        """Generate a rule-based greeting response."""
        question_lower = question.lower().strip()
        
        if any(greet in question_lower for greet in ['hi', 'hello', 'hey']):
            return (
                "Hello! I'm FloatChat, your AI assistant for exploring ARGO oceanographic data. "
                "I can help you find information about ocean temperature, salinity, and oceanographic profiles from around the world. "
                "Try asking me something like: 'Show me temperature data from the Atlantic Ocean' or 'What's the salinity like near Japan?'"
            )
        elif any(q in question_lower for q in ['what can you do', 'help', 'how does this work']):
            return (
                "I can help you explore ARGO oceanographic float data! Here's what I can do:\n\n"
                "🌊 Search for ocean temperature and salinity data by location\n"
                "📊 Find profiles from specific regions (Atlantic, Pacific, etc.)\n"
                "🗓️ Look up historical oceanographic measurements\n"
                "📍 Get data from specific coordinates\n\n"
                "Try asking: 'Show me recent temperature data from the Mediterranean' or 'What's the salinity profile near Australia?'"
            )
        else:
            return (
                "Hello! I'm FloatChat, here to help you explore oceanographic data. "
                "Ask me about ocean temperature, salinity, or specific regions you're interested in!"
            )

    @timing_decorator
    def query(self, user_question: str, top_k: int = None, include_metadata: bool = True) -> Dict:
        """
        Process a user query, routing to either conversational or RAG response.
        """
        if self.llm is None and self.openai_available:
            self.initialize_llm()

        # Inventory/listing route (handle before general conversation)
        if self._is_inventory_query(user_question):
            top_k = top_k or self.config.DEFAULT_TOP_K
            inv_profiles = self._get_inventory_profiles(top_k)
            if not inv_profiles:
                return {
                    'answer': "I couldn't list any ARGO profiles because none are loaded yet.",
                    'retrieved_profiles': [],
                    'query': user_question,
                    'success': False,
                    'query_type': 'inventory'
                }
            answer = self._generate_inventory_answer(inv_profiles)
            return {
                'answer': answer,
                'retrieved_profiles': self._format_retrieved_profiles(inv_profiles, include_metadata),
                'query': user_question,
                'context_profiles_count': len(inv_profiles),
                'success': True,
                'query_type': 'inventory'
            }

        # General info FAQs route (handle before small talk)
        info_intent = self._match_general_info_intent(user_question)
        if info_intent:
            answer = self._generate_general_info_answer(info_intent)
            return {
                'answer': answer,
                'retrieved_profiles': [],
                'query': user_question,
                'success': True,
                'query_type': 'general_info'
            }

        # Conversational route
        if self._is_general_conversation(user_question):
            logger.info("Identified general query, generating conversational response.")
            answer = self._generate_conversational_response(user_question)
            return {
                'answer': answer,
                'retrieved_profiles': [],
                'query': user_question,
                'success': True,
                'query_type': 'conversational'
            }

        # Data-specific RAG route
        top_k = top_k or self.config.DEFAULT_TOP_K
        logger.info(f"Retrieving top-{top_k} profiles for query: {user_question}")
        
        # Region availability guard: if the user asked for a specific ocean that's not present, return a clear message
        try:
            requested_ocean = self._detect_requested_ocean(user_question)
            if requested_ocean:
                oceans_available = self._available_oceans()
                if requested_ocean not in oceans_available:
                    parts: list[str] = []
                    parts.append(f"Summary: No local ARGO data was found for queries in the {requested_ocean}. ❌🌊")
                    if oceans_available:
                        parts.append(f"- Available locally: {', '.join(oceans_available)}")
                    else:
                        parts.append("- No oceans detected in the current dataset.")
                    parts.append("- This app uses your locally loaded NetCDF files; please load data for the requested ocean or adjust the region.")
                    parts.append("Next Step: Add ARGO files for the requested ocean, or ask about one of the available regions.")
                    return {
                        'answer': "\n".join(parts),
                        'retrieved_profiles': [],
                        'query': user_question,
                        'context_profiles_count': 0,
                        'success': False,
                        'query_type': 'region_unavailable'
                    }
        except Exception:
            pass
        # Intent-aware query boosting to improve retrieval quality
        ql = user_question.lower()
        boost_terms: list[str] = []

        # Hinglish/Indic synonyms mapping -> English domain terms
        sal_keywords = [
            "salinity", "psu", "practical salinity",
            "kharaapan", "kharapaan", "kharapan", "namak", "namkeen pani"
        ]
        temp_keywords = [
            "temperature", "temp", "°c", "taapman", "tapman", "garmi"
        ]
        depth_keywords = [
            "depth", "pressure", "dbar", "gehraai", "gehrayi", "gahrai"
        ]
        region_aliases = {
            "arabian sea": ["arab sagar", "arabi samudra", "arab samundar"],
            "bay of bengal": ["bangal ki khadi", "bengal ki khadi", "bangal khadi"],
            "equator": ["bhumadhya rekha", "equatorial"],
            "indian ocean": ["bharatiya mahāsāgar", "bhartiya mahasagar", "hind mahāsāgar", "hind mahasagar"],
        }

        if any(k in ql for k in sal_keywords):
            boost_terms += ["salinity", "psu", "practical salinity"]
        if any(k in ql for k in temp_keywords):
            boost_terms += ["temperature", "°c", "temp"]
        if any(k in ql for k in depth_keywords):
            boost_terms += ["pressure", "depth", "dbar"]
        for canon, aliases in region_aliases.items():
            if any(a in ql for a in aliases):
                boost_terms += [canon]
        # Boost with the requested high-level ocean if detected
        if requested_ocean:
            boost_terms.append(requested_ocean)
        boosted_query = user_question if not boost_terms else f"{user_question} | {' '.join(sorted(set(boost_terms)))}"

        retrieved_profiles = self.embedding_index.search_similar_profiles(boosted_query, top_k=top_k)

        # If a specific ocean was requested, prefer results from that ocean. If none, fall back to sampling from that ocean.
        if requested_ocean:
            def _meta_region(meta: Dict) -> str | None:
                try:
                    lat = meta.get('latitude'); lon = meta.get('longitude')
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        return self._region_from_coords(lat, lon)
                except Exception:
                    return None
                return None

            filtered = []
            for p in retrieved_profiles:
                reg = _meta_region(p.get('metadata', {}) or {}) or ""
                if requested_ocean.split()[0] in reg or requested_ocean in reg:
                    filtered.append(p)
            if filtered:
                retrieved_profiles = filtered
            else:
                # Fallback: sample profiles by ocean from the full metadata to build context
                sampled = self._get_profiles_by_ocean(requested_ocean, top_k)
                if sampled:
                    retrieved_profiles = sampled

        if not retrieved_profiles:
            # Build a helpful, data-aware message using available metadata
            metas = self.embedding_index.profile_metadata or []
            regions: list[str] = []
            times = []
            latitudes: list[float] = []
            longitudes: list[float] = []
            coord_samples: list[str] = []
            file_samples: list[str] = []
            temp_values: list[float] = []
            sal_values: list[float] = []
            depth_values: list[float] = []
            for m in metas:
                try:
                    lat = m.get('latitude'); lon = m.get('longitude')
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        regions.append(self._region_from_coords(lat, lon))
                        latitudes.append(lat)
                        longitudes.append(lon)
                        if len(coord_samples) < 5:
                            coord_samples.append(f"({lat:.2f}°, {lon:.2f}°)")
                    fs = m.get('file_source')
                    if fs and len(file_samples) < 5 and fs not in file_samples:
                        file_samples.append(str(fs))
                    # collect ranges if present
                    tr = m.get('temp_range')
                    if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                        temp_values.extend([tr[0], tr[1]])
                    sr = m.get('sal_range')
                    if isinstance(sr, (list, tuple)) and len(sr) >= 2:
                        sal_values.extend([sr[0], sr[1]])
                    dr = m.get('depth_range')
                    if isinstance(dr, (list, tuple)) and len(dr) >= 2:
                        depth_values.extend([dr[0], dr[1]])
                    if m.get('time'):
                        times.append(m['time'])
                except Exception:
                    pass
            regions = sorted(set([r for r in regions if r]))
            time_span = None
            if times:
                try:
                    tmin, tmax = min(times), max(times)
                    time_span = f"{self._format_month_year(tmin)} to {self._format_month_year(tmax)}"
                except Exception:
                    time_span = None

            bbox = None
            if latitudes and longitudes:
                try:
                    bbox = f"lat {min(latitudes):.2f}° to {max(latitudes):.2f}°, lon {min(longitudes):.2f}° to {max(longitudes):.2f}°"
                except Exception:
                    bbox = None

            tips = [
                "Try adding a region (e.g., 'Arabian Sea' or 'Bay of Bengal').",
                "Specify a time window (e.g., 'March 2023' or '2023-03').",
                "Mention the parameter explicitly (e.g., 'salinity', 'temperature').",
                "Increase Top K in the sidebar to broaden the search.",
            ]
            catalog = []
            total_profiles = len(metas)
            catalog.append(f"Available in RAG — total profiles: {total_profiles}")
            if regions:
                catalog.append(f"Available regions in the data: {', '.join(regions)}")
            if bbox:
                catalog.append(f"Coordinate coverage (bounding box): {bbox}")
            if coord_samples:
                catalog.append(f"Sample coordinates: {', '.join(coord_samples)}")
            if file_samples:
                catalog.append(f"Sample floats/files: {', '.join(file_samples)}")
            # parameter coverage
            param_flags = []
            if temp_values:
                try:
                    catalog.append(f"Temperature coverage: {min(temp_values):.1f}°C to {max(temp_values):.1f}°C")
                except Exception:
                    param_flags.append("temperature")
            else:
                param_flags.append("temperature")
            if sal_values:
                try:
                    catalog.append(f"Salinity coverage: {min(sal_values):.2f}–{max(sal_values):.2f} PSU")
                except Exception:
                    param_flags.append("salinity")
            else:
                param_flags.append("salinity")
            if depth_values:
                try:
                    catalog.append(f"Depth coverage: {min(depth_values):.0f}–{max(depth_values):.0f} m")
                except Exception:
                    pass
            if time_span:
                catalog.append(f"Time span covered: {time_span}")

            # Build a structured, emoji-friendly message
            lines = []
            lines.append("Summary: No exact matches for your query, but here’s what’s available in the local RAG dataset. 🌊")
            # Convert catalog entries to concise bullet points
            for item in catalog:
                lines.append(f"- 🔎 {item}")
            lines.append("📊 Visualization Suggestion: View a map of float locations within the bounding box, or plot depth–temperature/salinity profiles to explore coverage.")
            lines.append("🧭 Next Step: Refine by region, month/year, and parameter (temperature/salinity), or increase Top K to broaden search.")
            return {
                'answer': "\n".join(lines),
                'retrieved_profiles': [],
                'query': user_question,
                'success': False
            }

        # Apply MCP planning/filters if available
        mcp_plan = None
        try:
            retrieved_profiles, mcp_plan = self._apply_mcp_filters(retrieved_profiles, user_question)
        except Exception:
            pass

        context = self._build_context(retrieved_profiles)
        
        # Try LLM response first, fall back to rule-based
        if self.llm:
            answer = self._generate_llm_response(user_question, context)
        else:
            answer = self._generate_rule_based_response(user_question, retrieved_profiles)

        return {
            'answer': answer,
            'retrieved_profiles': self._format_retrieved_profiles(retrieved_profiles, include_metadata),
            'query': user_question,
            'context_profiles_count': len(retrieved_profiles),
            'success': True,
            'query_type': 'data_query',
            'mcp': mcp_plan  # include SQL/filters if generated
        }

    def _build_context(self, retrieved_profiles: List[Dict]) -> str:
        """Build context string from retrieved profiles, including region and month-year when possible."""
        lines: list[str] = []
        for i, profile in enumerate(retrieved_profiles):
            base = profile.get('summary', '')
            meta = profile.get('metadata', {}) or {}
            region = None
            timestr = None
            try:
                lat = meta.get('latitude'); lon = meta.get('longitude')
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    region = self._region_from_coords(lat, lon)
                t = meta.get('time')
                if t:
                    timestr = self._format_month_year(t)
            except Exception:
                pass
            suffix_parts = []
            if region:
                suffix_parts.append(f"Region: {region}")
            if timestr:
                suffix_parts.append(f"Time: {timestr}")
            suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
            lines.append(f"{i+1}. {base}{suffix}")
        return "\n".join(lines)

    def _generate_llm_response(self, question: str, context: str) -> str:
        """Generate a data-driven response using the OpenAI LLM."""
        available_oceans = ", ".join(self._available_oceans())
        system_prompt = (
            "You are FloatChat, an AI-powered multilingual assistant for exploring ARGO oceanographic data and answering general questions.\n"
            "You combine RAG with friendly explanations and clear recommendations.\n\n"
            "LANGUAGE SUPPORT:\n"
            "- Respond in the same language as the Human message (supports English, Hindi, Marathi, Tamil, Bengali, Telugu, Kannada, Gujarati, Punjabi).\n"
            "- Handle Hinglish and code-mixed Indian languages; recognize transliterated terms (e.g., 'taapman' ~ temperature, 'kharaapan/namak' ~ salinity, 'gehraai' ~ depth).\n"
            "- If mixed languages, default to English but keep key technical terms in the local language when helpful.\n\n"
            "OBJECTIVES:\n"
            "1. Grounding: Use retrieved ARGO float profile summaries first. Do not hallucinate values.\n"
            "2. Clarity: Provide a one-line summary first, then structured details.\n"
            "3. Multilingual Support: Keep technical terms in English if no common translation exists.\n"
            "4. General-Purpose Support: If no relevant context is found, answer with general knowledge and explicitly mention no local ARGO data was found.\n"
            "5. Next-Step Guidance: Suggest follow-up queries or visualizations.\n"
            "6. Transparency: If confidence is low, say so; if data is missing, suggest filters (region, time, parameter).\n\n"
            "LENGTH & STYLE:\n"
            "- Aim for around 7–8 sentences total across sections.\n"
            "- Include 2–4 relevant emojis to improve readability (e.g., 🌊, 📊, 🗺️, 🧭).\n\n"
            "RESPONSE STRUCTURE:\n"
            "- Summary: One clear line answering the query.\n"
            "- Details: 2–4 short bullet points citing numbers, regions, depths, times from the provided context.\n"
            "- Always include the named oceanic region when possible (e.g., Arabian Sea, Bay of Bengal, Equatorial Indian Ocean).\n"
            "- Format months as three-letter uppercase (JAN, FEB, …) with year (e.g., JAN 2023).\n"
            "- Next Step: Suggest how to refine or explore further.\n\n"
            "RULES:\n"
            "- Never make up numbers — rely on retrieved data.\n"
            "- Stay polite, concise, beginner-friendly.\n"
            "- If off-topic, answer intelligently and note it's general knowledge.\n\n"
            "DATA AVAILABILITY:\n"
            f"- AVAILABLE_OCEANS: {available_oceans if available_oceans else '(none detected)'}\n"
            "- If the question mentions an ocean that is NOT in AVAILABLE_OCEANS, clearly state that local data is unavailable for that ocean.\n"
            "- If the question mentions an ocean that IS in AVAILABLE_OCEANS but the CONTEXT snippet lacks that ocean, still answer helpfully, make it clear the local dataset contains that ocean generally, and suggest filters (region/time/parameter) to surface those profiles.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "USER: 'Show me salinity profiles near the equator in March 2023.'\n"
            "ASSISTANT:\n"
            "Summary: Salinity near the equator in March 2023 was stable and mid-range.\n"
            "- Profiles sampled between 0°–5° latitude, depth 0–2000 m.\n"
            "- Salinity range: 34.5–35.0 PSU across most profiles.\n"
            "- Concentration highest near 20°E–30°E longitude.\n\n"
            "Visualization Suggestion: Depth vs Salinity line plot.\n\n"
            "USER: 'अरब सागर में पिछले महीने का तापमान क्या था?'\n"
            "ASSISTANT:\n"
            "Summary: अरब सागर का औसत तापमान पिछले महीने मध्यम था।\n"
            "- सतह तापमान ~28°C–29°C।\n"
            "- गहराई के साथ तापमान धीरे-धीरे घटता है।\n"
            "- अधिकांश प्रोफाइल 0–2000 m गहराई तक।\n\n"
            "Visualization Suggestion: Depth–Temperature profile plot.\n\n"
            "USER: 'Who is the Prime Minister of India?'\n"
            "ASSISTANT: Summary: Currently, the Prime Minister of India is Shri Narendra Modi (as of 2025).\n"
            "(This answer is general knowledge, not based on ARGO data.)\n"
        )

        user_prompt = (
            "Use the following retrieved ARGO float profile summaries as primary context when relevant.\n\n"
            f"CONTEXT:\n{context if context else '(no local ARGO context found)'}\n\n"
            f"QUESTION: {question}\n\n"
            "Instructions:\n"
            "- Follow the RESPONSE STRUCTURE.\n"
            "- If CONTEXT is empty or irrelevant, answer using general knowledge and clearly state that no relevant local ARGO data was found for this query.\n"
            "- If confidence is low, say so and suggest better filters.\n"
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        try:
            # Try different invocation methods
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(messages)
            elif hasattr(self.llm, '__call__'):
                response = self.llm(messages)
            else:
                response = self.llm.predict([msg.content for msg in messages])
                return response
                
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_rule_based_response(question, [{'summary': context.split('\n')[0], 'metadata': {}}])

    def _generate_rule_based_response(self, question: str, retrieved_profiles: List[Dict]) -> str:
        """Generate a rule-based response using retrieved profiles."""
        if not retrieved_profiles:
            return (
                "Summary: No local ARGO profiles matched your query. 🙏\n"
                "- 💡 Details: Try adding a region (e.g., 'Arabian Sea' or 'Bay of Bengal'), a time window (e.g., 'March 2023'), and the parameter (salinity/temperature).\n"
                "- 🌊 Tip: Mention depth if relevant (e.g., surface vs 0–2000 m) to improve results.\n"
                "- 🧠 Note: I can still answer with general knowledge if you want; just say the word.\n"
                "📊 Visualization Suggestion: Ask for a map of float locations or a depth–temperature profile.\n"
                "🧭 Next Step: Increase Top K in the sidebar or refine filters (region, month/year, parameter).\n"
                "(This message is not based on local ARGO data.)"
            )
        
        num_profiles = len(retrieved_profiles)
        top_profile = retrieved_profiles[0]
        top_summary = top_profile['summary']
        
        # Extract statistics and metadata for display
        temperatures = []
        depths = []
        locations = []
        times: list = []
        primary_region: str | None = None
        
        for profile in retrieved_profiles:
            metadata = profile.get('metadata', {})
            
            # Collect temperature ranges
            if 'temp_range' in metadata and metadata['temp_range']:
                try:
                    temp_range = metadata['temp_range']
                    if isinstance(temp_range, (list, tuple)) and len(temp_range) >= 2:
                        temperatures.extend([temp_range[0], temp_range[1]])
                except:
                    pass
            
            # Collect depth ranges
            if 'depth_range' in metadata and metadata['depth_range']:
                try:
                    depth_range = metadata['depth_range']
                    if isinstance(depth_range, (list, tuple)) and len(depth_range) >= 2:
                        depths.extend([depth_range[0], depth_range[1]])
                except:
                    pass
            
            # Collect locations
            if 'latitude' in metadata and 'longitude' in metadata:
                locations.append((metadata['latitude'], metadata['longitude']))
                if primary_region is None:
                    try:
                        primary_region = self._region_from_coords(metadata['latitude'], metadata['longitude'])
                    except Exception:
                        primary_region = None
            # Collect times
            if metadata.get('time'):
                times.append(metadata['time'])
        
        # Build response (structured)
        lines: list[str] = []
        lines.append(f"Summary: Found {num_profiles} profile(s) relevant to your query. 🌊")
        lines.append(f"- 🔎 Details: Most relevant profile — {top_summary}")
        if temperatures:
            temp_min, temp_max = min(temperatures), max(temperatures)
            lines.append(f"- 🌡️ Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C across retrieved profiles")
        if depths:
            depth_min, depth_max = min(depths), max(depths)
            lines.append(f"- 📏 Depth coverage: {depth_min:.0f} m to {depth_max:.0f} m")
        if locations:
            lat, lon = locations[0]
            lines.append(f"- 📍 Primary location sample: {lat:.2f}°, {lon:.2f}°")
        if primary_region:
            lines.append(f"- 🗺️ Region: {primary_region}")
        if times:
            try:
                tmin, tmax = min(times), max(times)
                lines.append(f"- 🕒 Time span in results: {self._format_month_year(tmin)} to {self._format_month_year(tmax)}")
            except Exception:
                pass
        lines.append("📊 Visualization Suggestion: Depth vs Temperature/Salinity profile plot or T–S diagram.")
        lines.append("🧭 Next Step: Refine by region, month/year, and parameter; or ask to map float locations.")
        return "\n".join(lines)

    def _generate_fallback_response(self, question: str, retrieved_profiles: List[Dict]) -> str:
        """Generate a simple fallback response if LLM fails."""
        if not retrieved_profiles:
            return "No relevant oceanographic data found for your query."
        top_summary = retrieved_profiles[0]['summary']
        return (
            f"Based on the most relevant profile found: {top_summary}. "
            f"Found {len(retrieved_profiles)} relevant profiles matching your query."
        )

    def _format_retrieved_profiles(self, profiles: List[Dict], include_metadata: bool) -> List[Dict]:
        """Format retrieved profiles for API response."""
        formatted = []
        for profile in profiles:
            formatted_profile = {
                'summary': profile['summary'],
                'similarity_score': profile['similarity_score']
            }
            if include_metadata:
                metadata = profile.get('metadata', {})
                formatted_profile.update({
                    'latitude': metadata.get('latitude'),
                    'longitude': metadata.get('longitude'),
                    'time': metadata.get('time').isoformat() if metadata.get('time') else None,
                    'time_human': self._format_month_year(metadata.get('time')) if metadata.get('time') else None,
                    'file_source': metadata.get('file_source'),
                    'depth_range': metadata.get('depth_range'),
                    'temperature_range': metadata.get('temp_range'),
                    'region': (self._region_from_coords(metadata.get('latitude'), metadata.get('longitude'))
                               if isinstance(metadata.get('latitude'), (int, float)) and isinstance(metadata.get('longitude'), (int, float)) else None),
                })
            formatted.append(formatted_profile)
        return formatted