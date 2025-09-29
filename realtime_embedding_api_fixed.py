from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env at project root
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

import os
import json
import hashlib
import base64
import re
import logging
import sys
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import google.generativeai as genai
import requests
import tempfile
import asyncio
import numpy as np
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import traceback
# CLAP v10 Ultimate Analyzer
from zen_music_analyzer_v10_ultimate import ZenMusicAnalyzerV10Ultimate
zen_analyzer = ZenMusicAnalyzerV10Ultimate()

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Configure logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("EmbeddingAPI")

def log_separator(title):
    """Create a visual separator for major operations"""
    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info(f"[TARGET] {title}")
    logger.info(f"{separator}")

def log_step(step_num, total_steps, description):
    """Log a numbered step in a process"""
    logger.info(f"[STEP] Step {step_num}/{total_steps}: {description}")

def log_success(message):
    """Log successful operations"""
    logger.info(f"[SUCCESS] {message}")

def log_error(message, error=None):
    """Log errors with optional exception details"""
    error_msg = f"[ERROR] {message}"
    if error:
        error_msg += f" | Details: {str(error)}"
    logger.error(error_msg)

def log_warning(message):
    """Log warnings"""
    logger.warning(f"[WARNING] {message}")

def log_info(message):
    """Log general information"""
    logger.info(f"[INFO] {message}")

def log_debug(message):
    """Log debug information"""
    logger.debug(f"[DEBUG] {message}")

app = FastAPI()

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_info("FastAPI application initialized with CORS middleware")

# Supabase client
try:
    supabase: Client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    log_success("Supabase client initialized successfully")
except Exception as e:
    log_error("Failed to initialize Supabase client", e)

# Configure Google AI with enhanced authentication and fallback
auth_method = "unknown"
try:
    # Primary: Use Google AI Studio API (simpler and more stable)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Test the connection
        try:
            models = list(genai.list_models())
            gemini_models = [m for m in models if 'gemini-2.5-flash' in m.name]
            if gemini_models:
                auth_method = "google_ai_studio"
                log_success(f"Google AI Studio configured successfully - {len(models)} models available")
                log_info(f"Found {len(gemini_models)} Gemini 2.5 Flash variants")
            else:
                raise Exception("No Gemini 2.5 Flash models found")
        except Exception as test_error:
            log_warning(f"Google AI Studio test failed: {test_error}")
            raise test_error
    else:
        raise Exception("No GOOGLE_AI_API_KEY found")
        
except Exception as ai_studio_error:
    log_warning(f"Google AI Studio setup failed: {ai_studio_error}")
    log_info("Attempting Vertex AI fallback...")
    
    # Fallback: Use Vertex AI Service Account
    try:
        service_account_key = os.getenv('GEMINI_SERVICE_ACCOUNT_KEY')
        if service_account_key:
            import json
            from google.oauth2 import service_account
            import google.auth.transport.requests
            
            # Parse service account credentials
            credentials_info = json.loads(service_account_key.replace("'", '"'))
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Configure for Vertex AI
            auth_method = "vertex_ai_service_account"
            log_success("Vertex AI Service Account configured as fallback")
        else:
            raise Exception("No GEMINI_SERVICE_ACCOUNT_KEY found")
    except Exception as vertex_error:
        log_error("Both authentication methods failed", vertex_error)
        auth_method = "failed"

log_info(f"Final authentication method: {auth_method}")

def _get_google_access_token(service_account_info):
    """Get Google access token using service account (same as Supabase Edge Function)"""
    import time
    import jwt
    import requests
    
    GOOGLE_TOKEN_URI = 'https://oauth2.googleapis.com/token'
    GOOGLE_CLOUD_PLATFORM_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
    
    # Create JWT header
    header = {
        "alg": "RS256",
        "typ": "JWT"
    }
    
    # Create JWT claims
    now = int(time.time())
    claims = {
        "iss": service_account_info["client_email"],
        "sub": service_account_info["client_email"],
        "aud": GOOGLE_TOKEN_URI,
        "iat": now,
        "exp": now + 3600,
        "scope": GOOGLE_CLOUD_PLATFORM_SCOPE
    }
    
    # Create JWT token
    private_key = service_account_info["private_key"]
    token = jwt.encode(claims, private_key, algorithm="RS256", headers=header)
    
    # Exchange JWT for access token
    response = requests.post(GOOGLE_TOKEN_URI, data={
        'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion': token
    })
    
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Failed to get access token: {response.status_code} - {response.text}")

def get_auth_headers():
    """Get appropriate headers based on authentication method"""
    if auth_method == "vertex_ai_service_account":
        # Get fresh access token for Vertex AI
        credentials.refresh(google.auth.transport.requests.Request())
        return {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
    return None

def analyze_content_with_gemini_bolt(file_path, file_bytes, content_type):
    """Fast Bolt analysis - focused 3k character analysis for speed"""
    log_separator(f"BOLT {content_type.upper()} ANALYSIS")
    log_step(1, 4, f"Starting BOLT {content_type} analysis for: {file_path}")
    log_info(f"File size: {len(file_bytes)} bytes")

    # Focused analysis prompts - essential elements only
    if content_type == "video":
        analysis_prompt = """Analyze this video briefly. Focus on: visual style, main content, audio elements, mood, and sync licensing potential. Keep response under 3000 characters."""
    else:
        analysis_prompt = """Analyze this audio file with focus on sync licensing essentials. Provide a concise analysis covering:

**Genre & Style** (primary genre, subgenre, influences)
**Key Elements** (main instruments, production style, notable techniques)  
**Technical Details** (BPM, key signature, tempo feel)
**Mood & Energy** (emotional tone, intensity level, atmosphere)
**Sync Suitability** (what type of media this would work for - film/TV/ads/games)

Be specific but concise. Target around 2500-3000 characters total. Focus on the most important characteristics for music licensing decisions."""

    return _execute_gemini_analysis(file_path, file_bytes, content_type, analysis_prompt, "BOLT")

def _execute_gemini_analysis(file_path, file_bytes, content_type, analysis_prompt, mode):
    """Shared analysis execution logic for both Bolt and Deep modes"""
    log_step(2, 4, f"Generated {mode} {content_type} analysis prompt")

    # Use Vertex AI with proper authentication (same as working Supabase Edge Function)
    try:
        log_step(3, 4, f"Attempting Vertex AI analysis ({mode} mode)")
        
        # Get service account key from environment
        service_account_key = os.getenv('GEMINI_SERVICE_ACCOUNT_KEY')
        if not service_account_key:
            raise Exception("GEMINI_SERVICE_ACCOUNT_KEY not found")
        
        # Parse service account key
        import json
        service_account_info = json.loads(service_account_key)
        
        # Get access token using the same method as Supabase Edge Function
        access_token = _get_google_access_token(service_account_info)
        
        # Create temporary file
        file_extension = file_path.split('.')[-1].lower() if '.' in file_path else 'tmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        
        log_info(f"Created temporary file: {tmp_file_path}")
        
        try:
            # Encode file to base64
            with open(tmp_file_path, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine MIME type
            mime_type = f"audio/{file_extension}" if content_type == "audio" else f"video/{file_extension}"
            if file_extension == "mp3":
                mime_type = "audio/mpeg"
            elif file_extension == "mp4":
                mime_type = "video/mp4"
            elif file_extension == "wav":
                mime_type = "audio/wav"
            
            # Vertex AI API endpoint
            project_id = "mvpasap"  # Your project ID
            location = "us-central1"
            model_id = "gemini-2.5-flash"
            
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
            
            # Get auth headers
            headers = get_auth_headers()
            if not headers:
                raise Exception("No authentication headers available")
            
            # Request payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": analysis_prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": file_data
                                }
                            }
                        ]
                    }
                ]
            }
            
            log_info(f"Sending {content_type} file to Vertex AI for {mode} analysis...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    log_step(4, 4, f"Generating {mode} content analysis...")
                    log_success(f"{mode} {content_type.title()} analysis complete!")
                    log_info(f"Analysis length: {len(content)} characters")
                    log_info(f"Full analysis: {content}")
                    
                    return content
                else:
                    log_warning("Vertex AI returned no content")
            else:
                log_error(f"Vertex AI API error: {response.status_code} - {response.text}")
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
                log_info("Cleaned up temporary local file")
            except:
                pass
                
    except Exception as e:
        log_warning(f"Vertex AI {mode} analysis failed: {str(e)}")
        log_info("Falling back to Google AI Studio...")
    
    # Fallback to Google AI Studio
    try:
        log_step(3, 4, f"Attempting Google AI Studio fallback ({mode} mode)")
        
        # Initialize Gemini model for Google AI Studio
        model = genai.GenerativeModel('gemini-2.5-flash')
        log_info(f"Initialized Gemini model - gemini-2.5-flash")
        
        # Create temporary file
        import time
        import os
        
        file_extension = file_path.split('.')[-1].lower() if '.' in file_path else 'tmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        
        log_info(f"Created temporary file: {tmp_file_path}")
        
        try:
            # Upload file to Gemini
            uploaded_file = genai.upload_file(tmp_file_path)
            log_success(f"File uploaded successfully. Gemini file ID: {uploaded_file.name}")
            
            # Shorter wait time for better responsiveness
            wait_time = 5 if content_type == "audio" else 8
            log_info(f"Waiting {wait_time}s for file to become active...")
            time.sleep(wait_time)
            
            # Check file state
            file_info = genai.get_file(uploaded_file.name)
            log_info(f"File state check 1: {file_info.state.name}")
            
            if file_info.state.name == "ACTIVE":
                log_step(4, 4, f"Generating {mode} content analysis...")
                
                # Add timeout protection to prevent hanging
                try:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Gemini API call timed out after 60 seconds")
                    
                    # Set timeout for Gemini API call
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)  # 60 second timeout
                    
                    response = model.generate_content([analysis_prompt, uploaded_file])
                    
                    # Cancel timeout
                    signal.alarm(0)
                except TimeoutError as e:
                    log_warning(f"Gemini API call timed out: {e}")
                    response = None
                except Exception as e:
                    signal.alarm(0)  # Make sure to cancel timeout
                    raise e
                
                if response and response.text:
                    log_success(f"{mode} Analysis complete! Description length: {len(response.text)} chars")
                    log_info(f"Full analysis: {response.text}")
                    
                    # Clean up
                    try:
                        genai.delete_file(uploaded_file.name)
                        log_info("Cleaned up Gemini uploaded file")
                    except:
                        pass
                        
                    return result
                    
                    return response.text
                else:
                    log_warning("Gemini returned empty response")
                    
            else:
                log_warning(f"File not active (state: {file_info.state.name}), retrying...")
                # One more try with longer wait
                time.sleep(10)
                file_info = genai.get_file(uploaded_file.name)
                log_info(f"File state check 2: {file_info.state.name}")
                
                if file_info.state.name == "ACTIVE":
                    # Add timeout protection to prevent hanging
                    try:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Gemini API call timed out after 60 seconds")
                        
                        # Set timeout for Gemini API call
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60 second timeout
                        
                        response = model.generate_content([analysis_prompt, uploaded_file])
                        
                        # Cancel timeout
                        signal.alarm(0)
                    except TimeoutError as e:
                        log_warning(f"Gemini API call timed out: {e}")
                        response = None
                    except Exception as e:
                        signal.alarm(0)  # Make sure to cancel timeout
                        raise e
                    
                    if response and response.text:
                        log_success(f"{mode} Analysis complete! Description length: {len(response.text)} chars")
                        
                        # Clean up
                        try:
                            genai.delete_file(uploaded_file.name)
                            log_info("Cleaned up Gemini uploaded file")
                        except:
                            pass
                        
                        return response.text
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
                log_info("Cleaned up temporary local file")
            except:
                pass
                
    except Exception as e:
        log_error(f"Google AI Studio {mode} analysis failed", e)
    
    # Fallback to filename-based description
    log_warning(f"All {mode} analysis methods failed, using fallback description")
    return _create_fallback_description(file_path, content_type)

def analyze_content_with_gemini(file_path, file_bytes, content_type):
    """Deep Gemini analysis using Vertex AI - returns structured JSON response"""
    log_separator(f"DEEP {content_type.upper()} ANALYSIS")
    log_step(1, 4, f"Starting DEEP {content_type} analysis for: {file_path}")
    log_info(f"File size: {len(file_bytes)} bytes")

    # Request structured JSON output
    print(f"[DEBUG] Starting Gemini analysis for {content_type}")
    if content_type == "video":
        analysis_prompt = """Analyze this video file. Describe what you see and hear in detail."""
    else:
        analysis_prompt = """Analyze this audio file and provide a comprehensive musical analysis. Please include detailed information about:

**Genre and Subgenre Details:**
- Primary genre and specific subgenres
- Style characteristics and influences
- Era or time period associations

**Instrumentation Breakdown:**
- All instruments present (acoustic and electronic)
- Prominent vs supporting instruments
- Playing techniques and timbres
- Sound design elements

**Vocal & Lyric Analysis:**
- Vocal style, range, and characteristics
- Singing technique and delivery
- Lyrical themes and content (if audible)
- Vocal processing and effects

**Musical Structure Analysis:**
- Song sections (intro, verse, chorus, bridge, etc.)
- Arrangement and composition techniques
- Harmonic progressions and key changes
- Melodic patterns and motifs

**Tempo and Rhythm Analysis:**
- BPM estimation and tempo feel
- Time signature and rhythmic patterns
- Groove characteristics and swing
- Syncopation and rhythmic complexity

**Mood and Atmosphere:**
- Emotional tone and energy level
- Dynamic range and intensity
- Spatial qualities and ambiance
- Cultural or contextual associations

Please provide a thorough, detailed analysis with specific musical terminology and observations. Aim for comprehensive coverage of all audible musical elements."""

    return _execute_gemini_analysis(file_path, file_bytes, content_type, analysis_prompt, "DEEP")

def _analyze_with_vertex_ai(file_path, file_bytes, content_type, analysis_prompt):
    """Analyze content using Vertex AI (fallback method)"""
    log_info("Using Vertex AI fallback method")
    # Implementation for Vertex AI would go here
    # For now, return None to trigger metadata fallback
    return None

def _create_fallback_description(file_path, content_type):
    """Create a basic description from filename when Gemini fails"""
    import os
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    if content_type == "video":
        return f"Video content: {filename}. This is a video file containing audio and visual content."
    else:
        return f"Audio content: {filename}. This is an audio file suitable for music analysis and search."
    
async def analyze_audio_complete(audio_path,file_bytes):
    # Run CLAP and Gemini in parallel
    clap_task = asyncio.create_task( 
        asyncio.to_thread(zen_analyzer.analyze_audio_complete_sync, audio_path)
    )
    gemini_task = asyncio.create_task(
        asyncio.to_thread(analyze_content_with_gemini, audio_path, file_bytes, "audio")
    )

    # Wait for both to complete
    clap_result = await clap_task
    gemini_result = await gemini_task

    # Debug logging
    print(f"[DEBUG] CLAP result success: {clap_result.get('success')}")
    print(f"[DEBUG] CLAP result keys: {list(clap_result.keys()) if clap_result else 'None'}")
    if clap_result.get('success'):
        print(f"[DEBUG] CLAP BPM: {clap_result.get('bpm', {}).get('bpm')}")
        print(f"[DEBUG] CLAP Key: {clap_result.get('key', {}).get('key')}")
        print(f"[DEBUG] CLAP Energy: {clap_result.get('energy', {}).get('energy')}")
    else:
        print(f"[DEBUG] CLAP failed, using Gemini fallback")

    # Combine results - CLAP for technical, Gemini for creative
    clap_success = clap_result.get('success', False)
    
    # Parse Gemini result if it's a string (description)
    gemini_description = gemini_result if isinstance(gemini_result, str) else gemini_result.get('description', '')
    print(f"[DEBUG] Gemini result type: {type(gemini_result)}")
    print(f"[DEBUG] Gemini description: {gemini_description}")
    
    # Get genre from CLAP if available, otherwise use Gemini or default
    genre = "Unknown"
    if clap_success and clap_result.get('genre'):
        clap_genre = clap_result.get('genre')
        # Handle both string and object formats
        if isinstance(clap_genre, str):
            genre = clap_genre
        elif isinstance(clap_genre, dict) and clap_genre.get('genre'):
            genre = clap_genre.get('genre')
    elif isinstance(gemini_result, dict) and gemini_result.get('genre'):
        genre = gemini_result.get('genre')
    
    combined_result = {
        # From CLAP (technical parameters) - with better error handling
        "bpm": clap_result.get('bpm', {}).get('bpm') if clap_success else 120,
        "bpm_confidence": clap_result.get('bpm', {}).get('confidence') if clap_success else 0.1,
        "key": clap_result.get('key', {}).get('key') if clap_success else 'C major',
        "energy": clap_result.get('energy', {}).get('energy') if clap_success else 0.5,
        "tempo": clap_result.get('tempo', {}).get('tempo_class') if clap_success else 'Moderato',

        # From Gemini (creative parameters)
        "genre": genre,
        "mood": "",  # We'll extract this from the description if needed
        "description": gemini_description,

        # Metadata
        "analysis_method": "clap_v10_gemini_hybrid" if clap_success else "gemini_fallback",
        "clap_success": clap_success,
        "confidence_scores": {
            "bpm": clap_result.get('bpm', {}).get('confidence') if clap_success else 0.1,
            "key": clap_result.get('key', {}).get('confidence') if clap_success else 0.1,
            "energy": clap_result.get('energy', {}).get('confidence') if clap_success else 0.1
        }
    }

    print(f"[DEBUG] Combined result: {combined_result}")
    return combined_result

def create_weighted_description(text_desc, audio_desc, video_desc, tags):
    """Create weighted concatenated description based on your priority system"""
    log_separator("WEIGHTED DESCRIPTION CREATION")
    
    # Count available modalities
    modalities = []
    if text_desc: modalities.append("text")
    if audio_desc: modalities.append("audio") 
    if video_desc: modalities.append("video")
    
    log_info(f"Available modalities: {', '.join(modalities) if modalities else 'None'}")
    
    if not modalities:
        log_warning("No modalities available for description creation")
        return None
    
    log_step(1, 3, f"Creating weighted description from: {', '.join(modalities)}")
    
    # Add tags to all descriptions if provided
    tags_section = ""
    if tags and tags.strip():
        normalized_tags = normalize_tags(tags)
        tags_section = f"\n\nREQUIRED_TAGS: {', '.join(normalized_tags)}"
        log_info(f"Added normalized tags: {normalized_tags}")
    
    log_step(2, 3, "Determining weight distribution based on modality combination")
    
    # Determine weights based on available modalities
    final_description = None
    
    if len(modalities) == 1:
        # Single modality = 100%
        log_info("Single modality detected - using 100% weight")
        if text_desc: 
            final_description = f"SEARCH_INTENT: {text_desc}{tags_section}"
            log_info("Using text-only description")
        elif audio_desc: 
            final_description = f"AUDIO_CONTENT: {audio_desc}{tags_section}"
            log_info("Using audio-only description")
        elif video_desc: 
            final_description = f"VIDEO_CONTENT: {video_desc}{tags_section}"
            log_info("Using video-only description")
        
    elif len(modalities) == 2:
        log_info("Dual modality detected")
        if text_desc and audio_desc:
            # Text + Audio: 55% + 45%
            final_description = f"PRIMARY_INTENT: {text_desc}\n\nAUDIO_REFERENCE: {audio_desc}{tags_section}"
            log_info("Using Text(55%) + Audio(45%) weighting")
        elif text_desc and video_desc:
            # Text + Video: 70% + 30%  
            final_description = f"PRIMARY_INTENT: {text_desc}\n\nVIDEO_CONTEXT: {video_desc}{tags_section}"
            log_info("Using Text(70%) + Video(30%) weighting")
        elif audio_desc and video_desc:
            # Audio + Video: 65% + 35%
            final_description = f"AUDIO_CONTENT: {audio_desc}\n\nVIDEO_CONTEXT: {video_desc}{tags_section}"
            log_info("Using Audio(65%) + Video(35%) weighting")
            
    else:
        # All three: Text 45% + Audio 35% + Video 20%
        log_info("Triple modality detected - using Text(45%) + Audio(35%) + Video(20%) weighting")
        final_description = f"SEARCH_INTENT: {text_desc}\n\nAUDIO_REFERENCE: {audio_desc}\n\nVIDEO_CONTEXT: {video_desc}{tags_section}"
    
    log_step(3, 3, "Weighted description created successfully")
    log_info(f"Final description length: {len(final_description)} characters")
    log_debug(f"Final weighted description: {final_description}")
    
    return final_description

def normalize_tags(tags_string):
    """Normalize tags for semantic matching with enhanced JSON parsing"""
    if not tags_string:
        return []
    
    log_debug(f"Raw tags input: {repr(tags_string)}")
    
    # Handle different input formats
    tag_list = []
    
    # Try to parse as JSON array first (e.g., '["Electronic", "C Minor"]')
    if tags_string.strip().startswith('[') and tags_string.strip().endswith(']'):
        try:
            import json
            parsed_tags = json.loads(tags_string)
            tag_list = [str(tag).strip().lower() for tag in parsed_tags if str(tag).strip()]
            log_debug(f"Parsed as JSON array: {tag_list}")
        except json.JSONDecodeError:
            log_warning(f"Failed to parse JSON tags: {tags_string}")
            # Fallback to manual parsing
            content = tags_string.strip()[1:-1]  # Remove brackets
            tag_list = [tag.strip(' "\'').lower() for tag in content.split(',') if tag.strip(' "\'')]
            log_debug(f"Manual parsing fallback: {tag_list}")
    else:
        # Split by comma and clean (original behavior)
        tag_list = [tag.strip().lower() for tag in tags_string.split(',') if tag.strip()]
        log_debug(f"Parsed as comma-separated: {tag_list}")
    
    # Normalize common musical terms
    normalized = []
    for tag in tag_list:
        # Clean any remaining quotes
        tag = tag.strip(' "\'\t\n')
        
        # Handle key signatures
        tag = re.sub(r'\b([a-g])\s*min(or)?\b', r'\1 minor', tag)
        tag = re.sub(r'\b([a-g])\s*maj(or)?\b', r'\1 major', tag) 
        tag = re.sub(r'\b([a-g])m\b', r'\1 minor', tag)
        
        # Handle common abbreviations
        tag = tag.replace('bpm', 'beats per minute')
        tag = tag.replace('edm', 'electronic dance music')
        
        if tag:  # Only add non-empty tags
            normalized.append(tag)
            log_debug(f"Normalized tag: '{tag}'")
    
    log_info(f"Final normalized tags: {normalized}")
    return normalized

def create_narrative_summary(analysis_text, content_type):
    """Create a 200-300 character narrative summary from detailed analysis"""
    if not analysis_text or len(analysis_text) < 50:
        if content_type == "audio":
            return "An audio track with musical content suitable for listening applications."
        else:
            return "A video file containing audio and visual content for multimedia applications."
    
    # Extract key information from the analysis
    lines = analysis_text.split('\n')
    key_info = []
    
    # Look for genre, mood, tempo, and description information
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['genre', 'style', 'mood', 'tempo', 'bpm', 'atmosphere', 'energy']):
            # Clean up the line
            if ':' in line:
                value = line.split(':', 1)[1].strip()
                if value and not value.startswith('*'):
                    key_info.append(value)
    
    # If we have key info, create a narrative
    if key_info:
        # Take first few pieces of key info
        summary_parts = key_info[:3]
        narrative = f"This {content_type} features {', '.join(summary_parts[:2])}"
        if len(summary_parts) > 2:
            narrative += f" with {summary_parts[2]}"
        narrative += "."
    else:
        # Fallback: extract first meaningful sentence
        sentences = [s.strip() for s in analysis_text.replace('.', '.\n').split('\n') if s.strip()]
        meaningful_sentences = [s for s in sentences if len(s) > 20 and not s.startswith('*')]
        
        if meaningful_sentences:
            first_sentence = meaningful_sentences[0]
            # Truncate if too long
            if len(first_sentence) > 250:
                first_sentence = first_sentence[:247] + "..."
            narrative = first_sentence
        else:
            # Final fallback
            if content_type == "audio":
                narrative = "A musical audio track with distinctive characteristics and artistic elements."
            else:
                narrative = "A video presentation with visual and audio elements creating an engaging experience."
    
    # Ensure length is within bounds (200-300 chars)
    if len(narrative) < 200:
        # Pad with additional context if available
        if content_type == "audio":
            narrative += " The composition showcases unique musical elements and production techniques."
        else:
            narrative += " The visual presentation combines compelling imagery with complementary audio elements."
    
    # Truncate if too long
    if len(narrative) > 300:
        narrative = narrative[:297] + "..."
    
    return narrative

async def store_search_upload(title, description, embedding, content_type):
    """Store uploaded search file in search_index table"""
    try:
        log_info(f"[INFO] Storing search upload: {title}")
        
        # Prepare the data - store title in content field
        upload_data = {
            "content": title,
            "description": description,
            "embedding": embedding,
            "content_type": content_type,
            "content_id": str(uuid.uuid4()),  # Generate a unique ID for this content
            "search_text": title,  # Use title as searchable text
            "created_at": datetime.now().isoformat()
        }
        
        # Insert into search_index table
        response = supabase.table("search_index").insert(upload_data).execute()
        
        if response.data:
            log_success(f"[SUCCESS] Stored search upload: {title}")
            return True
        else:
            log_error(f"[ERROR] Failed to store search upload: {title}")
            return False
            
    except Exception as e:
        log_error(f"[ERROR] Error storing search upload {title}: {str(e)}")
        return False

def generate_embedding_for_file(file_path, file_bytes, text=None, tags=None):
    print(f"[INFO] Processing file: {file_path}, size: {len(file_bytes) if file_bytes else 0} bytes")
    
    # If no file provided (text-only search), just create description
    if not file_bytes:
        description = text or "Search query"
        if tags:
            description += f" Tags: {tags}"
    else:
        # Determine content type
        if file_path.lower().endswith(('.mp3', '.wav', '.m4a', '.aac')):
            content_type = "audio"
        elif file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            content_type = "video"
        else:
            content_type = "audio"  # Default
        
        if content_type == "audio":
            print(f" Running hybrid CLAP + Gemini analysis...")
            try:
                result = asyncio.run(analyze_audio_complete(file_path, file_bytes))
                print(f"[SUCCESS] Hybrid analysis result: {result}")
                description = json.dumps(result)  # store full hybrid output
            except Exception as e:
                print(f"[ERROR] Hybrid analysis failed: {str(e)}")
                description = f"Audio file {os.path.basename(file_path)} (fallback description)"
        else:
           print(f"🎥 Analyzing {content_type} content with Gemini...")


        try:
            # Use Google AI Studio for BOTH audio and video analysis
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Create content-specific analysis prompt
            if content_type == "video":
                analysis_prompt = f"""Analyze this video file comprehensively and provide a detailed description including:

**Visual Content:**
- What is happening in the video (scenes, actions, people, objects)
- Visual style and aesthetics
- Colors, lighting, composition
- Any text or graphics visible

**Audio Content:**
- Music genre, style, and characteristics
- Instruments and vocals present
- Tempo, rhythm, and musical structure
- Audio quality and production style

**Overall Content:**
- Main theme or subject matter
- Mood and atmosphere
- Target audience or purpose
- Cultural context or setting

Focus on creating a rich, searchable description that captures both the visual and audio elements of this video content."""
            else:
                analysis_prompt = f"""Analyze this audio file comprehensively and describe:

**Musical Characteristics:**
- Genre and subgenre
- Tempo (BPM estimate) and rhythm
- Key signature and chord progressions
- Melody and harmony structure

**Instrumentation and Production:**
- Primary and secondary instruments
- Vocal characteristics and style
- Production techniques and audio quality
- Mix and mastering characteristics

**Mood and Style:**
- Emotional tone and energy level
- Atmosphere and ambiance
- Musical influences and era
- Cultural context

Provide a detailed, descriptive analysis that captures the essence and unique characteristics of this musical content."""

            # Add user context if provided
            if text:
                analysis_prompt += f"\n\nAdditional context: {text}"
            if tags:
                analysis_prompt += f"\nTags: {tags}"

            # Upload file and analyze with proper error handling
            print(f"🤖 Sending {content_type} file to Gemini for analysis...")
            
            # Create temporary file for upload
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_path.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            uploaded_file = None
            description = None  # Initialize description to avoid scope issues
            
            try:
                # Upload file to Gemini with retry logic
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        print(f"📤 Upload attempt {attempt + 1}/{max_retries}...")
                        uploaded_file = genai.upload_file(tmp_file_path)
                        
                        # Wait for file to become active with longer timeout for videos
                        import time
                        wait_time = 10 if content_type == "video" else 5
                        print(f"⏳ Waiting {wait_time}s for file to become active...")
                        time.sleep(wait_time)
                        
                        # Check file state with multiple attempts
                        max_state_checks = 6
                        for state_check in range(max_state_checks):
                            file_info = genai.get_file(uploaded_file.name)
                            print(f"📁 File state check {state_check + 1}: {file_info.state.name}")
                            
                            if file_info.state.name == "ACTIVE":
                                # Generate content analysis
                                print("[TARGET] Analyzing content with Gemini...")
                                response = model.generate_content([analysis_prompt, uploaded_file])
                                
                                if response.text:
                                    description = response.text
                                    print(f"[SUCCESS] Analysis complete! Description length: {len(description)} chars")
                                    print(f"[INFO] Full analysis: {description}")
                                    success = True
                                    break
                                else:
                                    print("[WARNING] No analysis text returned")
                                    break
                                    
                            elif file_info.state.name == "FAILED":
                                print(f"[ERROR] File processing failed on Google's side")
                                break
                            else:
                                print(f"[WARNING] File still {file_info.state.name}, waiting 3s...")
                                time.sleep(3)
                        
                        if success:
                            break
                        else:
                            print(f"🔄 File never became active, trying next attempt...")
                            
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[ERROR] Attempt {attempt + 1} failed: {error_msg}")
                        
                        # Handle specific error types for debugging
                        if "404" in error_msg and "not found" in error_msg.lower():
                            print("[DEBUG] Model access issue - using fallback")
                            break
                        elif "400" in error_msg and "not in an ACTIVE state" in error_msg:
                            print("⏳ File state issue - retrying with longer wait")
                            time.sleep(5)
                        else:
                            time.sleep(2)
                        
                        if attempt == max_retries - 1:
                            print("🚫 All retry attempts exhausted")
                
                # Clean up uploaded file
                if uploaded_file:
                    try:
                        genai.delete_file(uploaded_file.name)
                        print("🗑️ Cleaned up uploaded file")
                    except:
                        print("[WARNING] Could not delete uploaded file")
                        pass
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
            
        except Exception as e:
            print(f"[ERROR] Gemini analysis failed: {str(e)}")
            description = None  # Will be handled below
        
        # Ensure description is always defined (fix for scope bug)
        if description is None:
            print("[INFO] Using fallback analysis due to processing issues...")
            # Create meaningful fallback description
            description = f"This {content_type} file"
            
            if file_path:
                # Extract meaningful info from filename
                filename_parts = file_path.replace('_', ' ').replace('-', ' ').split('.')
                clean_name = filename_parts[0].strip()
                if clean_name and clean_name not in ["videoplayback", "test-video", "test-audio"]:
                    description += f" '{clean_name}'"
            
            description += f" contains {content_type} content suitable for multimedia applications."
            
            if text:
                description += f" Context: {text}."
            if tags:
                description += f" Tags: {tags}."
                
            print(f"[INFO] Fallback description: {description}")
    
    # Now generate embedding from the description using Google AI
    print("🔄 Generating semantic embedding from analysis...")
    
    try:
        # Use embedding model via Google AI Studio
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=description
        )
        
        if embedding_result['embedding']:
            embedding = embedding_result['embedding']
            print(f"[SUCCESS] Generated embedding: {len(embedding)} dims, first 5: {embedding[:5]}")
            return embedding
        else:
            print("[ERROR] No embedding returned")
            return None
            
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {str(e)}")
        return None

@app.post("/api/generate-embedding")
async def generate_embedding(
    file: UploadFile = File(...),
    text: str = Form(None),
    tags: str = Form(None)
):
    print(f"\n[TARGET] === NEW REQUEST: {file.filename} ===")
    
    # Read file
    file_bytes = await file.read()
    file_path = file.filename
    
    print(f"[INFO] File size: {len(file_bytes)} bytes")
    print(f"[INFO] Text: {text}")
    print(f"[INFO] Tags: {tags}")

    # Generate embedding
    embedding = generate_embedding_for_file(file_path, file_bytes, text, tags)

    if embedding and len(embedding) == 768:
        try:
            # Convert embedding to PostgreSQL array format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            print(f"[INFO] Storing in database...")
            
            # Use raw SQL to ensure proper vector insertion
            result = supabase.rpc('insert_embedding', {
                'p_content': file_path,
                'p_embedding': embedding_str,
                'p_gemini_analysis': json.dumps({
                    'filename': file_path,
                    'size': len(file_bytes),
                    'text': text,
                    'tags': tags.split(',') if tags else []
                }),
                'p_gemini_mood': text if text else None,
                'p_gemini_tags': tags.split(',') if tags else None
            }).execute()
            
            print(f"[SUCCESS] SUCCESS! Stored embedding for {file_path}")
            return {
                "success": True, 
                "message": "Embedding generated and stored.",
                "embedding_dim": len(embedding)
            }
        except Exception as e:
            print(f"[ERROR] Database error: {str(e)}")
            return {"success": False, "error": str(e)}
    else:
        print(f"[ERROR] Failed to generate embedding")
        return {"success": False, "error": "Failed to generate embedding."}

@app.post("/api/gemini-embedding")
async def gemini_embedding_for_search(
    query: str = Form(None),
    tags: str = Form(None),
    audio: UploadFile = File(None),
    video: UploadFile = File(None)
):
    """Advanced multimodal search endpoint with weighted concatenation"""
    print(f"\n[DEBUG] === MULTIMODAL SEARCH REQUEST ===")
    print(f"Query: {query}")
    print(f"Tags: {tags}")
    print(f"Audio: {audio.filename if audio else None}")
    print(f"Video: {video.filename if video else None}")
    
    # Step 1: Process each modality separately
    text_description = None
    audio_description = None
    video_description = None
    
    # Process text query
    if query and query.strip():
        text_description = query.strip()
        print(f"[INFO] Text input: {text_description}")
    
    # Process audio file
    if audio:
        audio_bytes = await audio.read()
        print(f"[INFO] Processing audio: {audio.filename}")
        audio_description = analyze_content_with_gemini(audio.filename, audio_bytes, "audio")
        if audio_description:
            print(f"[INFO] Audio analysis: {audio_description}")
    
    # Process video file
    if video:
        video_bytes = await video.read()
        print(f"[INFO] Processing video: {video.filename}")
        video_description = analyze_content_with_gemini(video.filename, video_bytes, "video")
        if video_description:
            print(f"[INFO] Video analysis: {video_description}")
    
    # Step 2: Create weighted concatenated description
    combined_description = create_weighted_description(
        text_description, audio_description, video_description, tags
    )
    
    if not combined_description:
        print("[ERROR] No valid input provided")
        return {"error": "No valid input provided"}
    
    print(f"[TARGET] Final weighted description: {combined_description}")
    
    # Step 3: Generate single embedding from combined description
    try:
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=combined_description
        )
        
        if embedding_result['embedding']:
            embedding = embedding_result['embedding']
            print(f"[SUCCESS] Generated unified embedding: {len(embedding)} dims")
            return {"embedding": embedding, "description": combined_description}
        else:
            print("[ERROR] No embedding returned")
            return {"error": "Failed to generate embedding"}
            
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {str(e)}")
        return {"error": str(e)}

@app.post("/api/update-track-embedding")
async def update_track_embedding(
    track_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(None)
):
    """
    Endpoint to generate and update embedding for a specific track in the tracks table.
    Used when tracks are uploaded to the music library.
    """
    print(f"\n[INFO] === UPDATE TRACK EMBEDDING ===")
    print(f"[INFO] Track ID: {track_id}")
    print(f"[INFO] Title: {title}")
    print(f"[INFO] Description: {description}")
    
    # Create prompt from track metadata
    prompt = title
    if description:
        prompt += f" {description}"
    
    try:
        # Run hybrid CLAP + Gemini analysis for this track
        # Use the actual audio file from storage
        # Download the audio file from Supabase Storage
        from supabase import create_client
        import requests
        # Get the track record from DB
        track = supabase.table('tracks').select('audio_url').eq('id', track_id).single().execute()
        audio_url = None
        if track and track.data and 'audio_url' in track.data:
            audio_url = track.data['audio_url']
        if not audio_url:
            print(f"[ERROR] No audio_url found for track {track_id}")
            return {"success": False, "error": "No audio_url found for track"}
        # Download audio file
        try:
            print(f"[DEBUG] Attempting to download audio file for CLAP analysis: {audio_url}")
            response = requests.get(audio_url)
            response.raise_for_status()
            file_bytes = response.content
            print(f"[DEBUG] Downloaded {len(file_bytes)} bytes from {audio_url}")
            # Save to temp file for CLAP
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(file_bytes)
                audio_path = tmp_file.name
            print(f"[DEBUG] Saved audio file to temp path: {audio_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download audio file: {e}")
            return {"success": False, "error": "Failed to download audio file"}
        # Run analysis on actual audio file
        print(f"[DEBUG] Running CLAP+Gemini analysis on: {audio_path}")
        print(f"[DEBUG] Audio file size: {len(file_bytes)} bytes")
        print(f"[DEBUG] Audio file path: {audio_path}")
        
        # Check if file exists and is readable
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"[DEBUG] Temp file exists, size: {file_size} bytes")
        else:
            print(f"[ERROR] Temp file does not exist: {audio_path}")
            
        # Validate audio file before analysis
        try:
            import librosa
            print(f"[DEBUG] Validating audio file with librosa...")
            test_y, test_sr = librosa.load(audio_path, sr=22050, duration=5)  # Test with just 5 seconds
            print(f"[DEBUG] Audio file validation successful: {len(test_y)} samples, {test_sr} Hz")
        except Exception as e:
            print(f"[ERROR] Audio file validation failed: {e}")
            print(f"[ERROR] This audio file is corrupted or invalid. CLAP analysis will fail.")
            # Return a fallback result
            return {
                "bpm": 120,
                "bpm_confidence": 0.1,
                "key": "C major",
                "energy": 0.5,
                "tempo": "Moderato",
                "genre": "Unknown",
                "mood": "",
                "description": "Audio file validation failed - file may be corrupted",
                "analysis_method": "audio_validation_failed",
                "clap_success": False,
                "confidence_scores": {
                    "bpm": 0.1,
                    "key": 0.1,
                    "energy": 0.1
                }
            }
            
        analysis_result = await analyze_audio_complete(audio_path, file_bytes)
        print(f"[DEBUG] CLAP+Gemini analysis result: {analysis_result}")
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
            print(f"[DEBUG] Cleaned up temp file: {audio_path}")
        except Exception as e:
            print(f"[WARNING] Could not clean up temp file: {e}")

        # Generate embedding using the analysis result description
        description = analysis_result.get('description', '')
        if not description:
            description = f"Audio track: {title}"
        
        try:
            # Generate embedding using Gemini text-embedding model
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=description
            )
            embedding = response['embedding']
            print(f"[DEBUG] Generated embedding: {len(embedding)} dimensions")
        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            embedding = None

        # Prepare update fields - only include fields that exist in the database
        # Convert BPM to integer since database expects integer type
        bpm_value = analysis_result.get('bpm')
        print(f"[DEBUG] Original BPM value: {bpm_value} (type: {type(bpm_value)})")
        
        if bpm_value is not None:
            try:
                bpm_value = int(round(float(bpm_value)))
                print(f"[DEBUG] Converted BPM value: {bpm_value} (type: {type(bpm_value)})")
            except (ValueError, TypeError) as e:
                print(f"[ERROR] Failed to convert BPM: {e}")
                bpm_value = None
        
        # Convert genre to array format if it's a string
        genre_value = analysis_result.get('genre')
        print(f"[DEBUG] Original genre value: {genre_value} (type: {type(genre_value)})")
        
        if genre_value and isinstance(genre_value, str):
            genre_value = [genre_value]  # Convert string to array
            print(f"[DEBUG] Converted genre to array: {genre_value}")
        elif genre_value and isinstance(genre_value, list):
            genre_value = genre_value  # Already an array
            print(f"[DEBUG] Genre already array: {genre_value}")
        elif not genre_value:
            genre_value = ["Unknown"]  # Default to array with "Unknown"
            print(f"[DEBUG] Using default genre: {genre_value}")
        
        # Convert energy to float if it's a number, otherwise set to None
        energy_value = analysis_result.get('energy')
        if energy_value is not None:
            try:
                energy_value = float(energy_value)
            except (ValueError, TypeError):
                energy_value = None
        
        update_fields = {
            'embedding': embedding,
            'bpm': bpm_value,
            'bpm_confidence': analysis_result.get('bpm_confidence'),
            'key': analysis_result.get('key'),
            'energy': energy_value,
            'tempo': analysis_result.get('tempo'),
            'genre': genre_value,
            'description': analysis_result.get('description'),
            'analysis_method': analysis_result.get('analysis_method'),
            'clap_success': analysis_result.get('clap_success'),
            'confidence_scores': json.dumps(analysis_result.get('confidence_scores', {})),
        }
        
        print(f"[DEBUG] Update fields: {update_fields}")

        # Update the track in the database
        update_result = supabase.table('tracks').update(update_fields).eq('id', track_id).execute()

        if update_result.data:
            print(f"[SUCCESS] Updated track {track_id} with embedding and analysis fields")
            # Return the analysis results in the expected format
            return {
                "bpm": analysis_result.get('bpm'),
                "bpm_confidence": analysis_result.get('bpm_confidence'),
                "key": analysis_result.get('key'),
                "energy": analysis_result.get('energy'),
                "tempo": analysis_result.get('tempo'),
                "genre": analysis_result.get('genre'),
                "mood": analysis_result.get('mood'),
                "description": analysis_result.get('description'),
                "analysis_method": analysis_result.get('analysis_method'),
                "clap_success": analysis_result.get('clap_success'),
                "confidence_scores": analysis_result.get('confidence_scores'),
            }
        else:
            print(f"[ERROR] Failed to update track {track_id}")
            return {"success": False, "error": "Failed to update track in database"}
        
            
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/search")
async def advanced_search(
    query: str = Form(None),
    tags: str = Form(None),
    audio: UploadFile = File(None),
    video: UploadFile = File(None),
    match_threshold: float = Form(0.1),
    match_count: int = Form(10),
    search_mode: str = Form("deep"),  # "bolt" or "deep"
    precomputed_audio_analysis: str = Form(None),  # NEW: Precomputed audio analysis
    precomputed_video_analysis: str = Form(None)   # NEW: Precomputed video analysis
):
    """Advanced search with tag filtering as hard filters, then vector similarity"""
    log_separator("ADVANCED MULTIMODAL SEARCH")
    log_info(f"Search parameters:")
    log_info(f"  - Query: {query}")
    log_info(f"  - Tags: {tags}")
    log_info(f"  - Audio: {audio.filename if audio else None} ({audio.size if audio else 0} bytes)")
    log_info(f"  - Video: {video.filename if video else None} ({video.size if video else 0} bytes)")
    log_info(f"  - Match threshold: {match_threshold}")
    log_info(f"  - Match count: {match_count}")
    log_info(f"  - Search mode: {search_mode.upper()}")
    
    # Step 1: Generate embedding for multimodal search
    log_step(1, 5, f"Processing multimodal inputs ({search_mode.upper()} mode)")
    text_description = None
    audio_description = None 
    video_description = None
    
    if query and query.strip():
        text_description = query.strip()
        log_info(f"Text input processed: '{text_description[:50]}...'")
    
    # OPTIMIZED: Use precomputed analyses when available, or process files in parallel
    import asyncio
    
    async def process_audio():
        try:
            # Check for precomputed analysis first
            if precomputed_audio_analysis:
                log_info(f"Using precomputed audio analysis ({len(precomputed_audio_analysis)} chars)")
                return precomputed_audio_analysis
            elif audio:
                log_info(f"Processing audio file: {audio.filename} ({search_mode.upper()} mode)")
                analysis_function = analyze_content_with_gemini_bolt if search_mode.lower() == "bolt" else analyze_content_with_gemini
                audio_bytes = await audio.read()
                result = analysis_function(audio.filename, audio_bytes, "audio")
                if result:
                    log_success(f"{search_mode.upper()} audio analysis completed: {len(result)} chars")
                    # Generate summary immediately after analysis
                    summary = create_analysis_summary(result, "audio")
                    log_info(f"Audio summary generated: {summary[:100] if summary else 'None'}")
                else:
                    log_warning("Audio analysis failed - using fallback")
                return result
            return None
        except Exception as e:
            log_error(f"Error in process_audio: {str(e)}")
            return None
    
    async def process_video():
        # Check for precomputed analysis first
        if precomputed_video_analysis:
            log_info(f"Using precomputed video analysis ({len(precomputed_video_analysis)} chars)")
            return precomputed_video_analysis
        elif video:
            log_info(f"Processing video file: {video.filename} ({search_mode.upper()} mode)")
            analysis_function = analyze_content_with_gemini_bolt if search_mode.lower() == "bolt" else analyze_content_with_gemini
            video_bytes = await video.read()
            result = analysis_function(video.filename, video_bytes, "video")
            if result:
                log_success(f"{search_mode.upper()} video analysis completed: {len(result)} chars")
            else:
                log_warning("Video analysis failed - using fallback")
            return result
        return None
    
    # Run audio and video processing in parallel (or use precomputed)
    audio_description, video_description = await asyncio.gather(
        process_audio(),
        process_video(),
        return_exceptions=True
    )
    
    # Handle any exceptions from parallel processing
    if isinstance(audio_description, Exception):
        log_warning(f"Audio processing failed: {audio_description}")
        audio_description = None
    if isinstance(video_description, Exception):
        log_warning(f"Video processing failed: {video_description}")
        video_description = None
    
    # Step 2: Create weighted description (without tags - they're handled as filters)
    log_step(2, 5, "Creating weighted multimodal description")
    combined_description = create_weighted_description(
        text_description, audio_description, video_description, None
    )
    
    if not combined_description:
        log_error("No valid search input provided")
        return {"error": "No valid search input provided"}
    
    # Generate embedding for vector search
    log_step(3, 5, "Generating search embedding")
    try:
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=combined_description
        )
        
        if not embedding_result['embedding']:
            log_error("Failed to generate search embedding")
            return {"error": "Failed to generate search embedding"}
        
        query_embedding = embedding_result['embedding']
        log_success(f"Generated search embedding: {len(query_embedding)} dimensions")
        log_debug(f"First 5 embedding values: {query_embedding[:5]}")
        
    except Exception as e:
        log_error("Embedding generation failed", e)
        return {"error": str(e)}
    
    # Step 2: Apply tag filtering as hard filters
    log_step(4, 5, "Processing tag filters and search execution")
    normalized_tags = normalize_tags(tags) if tags else []
    
    try:
        if normalized_tags:
            log_info(f"Applying hard tag filters: {normalized_tags}")
            
            # Use proper Supabase filter methods for tag filtering 
            # This approach works better with array fields
            query_builder = supabase.table('tracks').select('*')
            
            # Process each tag and collect results
            tag_matched = False
            all_tag_results = []  # Store results for each tag separately
            
            for tag in normalized_tags:
                log_debug(f"Searching for tag: '{tag}'")
                
                # Try each search type independently for this tag
                field_results = []
                
                # 1. Array contains for exact match (genre, moods, instruments)
                try:
                    result1 = supabase.table('tracks').select('*').filter('genre', 'cs', f'{{{tag}}}').execute()
                    field_results.append(result1.data or [])
                except Exception as e:
                    log_debug(f"Genre search failed for '{tag}': {e}")
                    field_results.append([])
                
                try:
                    result2 = supabase.table('tracks').select('*').filter('moods', 'cs', f'{{{tag}}}').execute() 
                    field_results.append(result2.data or [])
                except Exception as e:
                    log_debug(f"Moods search failed for '{tag}': {e}")
                    field_results.append([])
                
                try:
                    result3 = supabase.table('tracks').select('*').filter('instruments', 'cs', f'{{{tag}}}').execute()
                    field_results.append(result3.data or [])
                except Exception as e:
                    log_debug(f"Instruments search failed for '{tag}': {e}")
                    field_results.append([])
                
                # 2. Text field fuzzy matches
                try:
                    result4 = supabase.table('tracks').select('*').filter('key', 'ilike', f'%{tag}%').execute()
                    field_results.append(result4.data or [])
                except Exception as e:
                    log_debug(f"Key search failed for '{tag}': {e}")
                    field_results.append([])
                
                try:
                    result5 = supabase.table('tracks').select('*').filter('vocal_type', 'ilike', f'%{tag}%').execute()
                    field_results.append(result5.data or [])
                except Exception as e:
                    log_debug(f"Vocal type search failed for '{tag}': {e}")
                    field_results.append([])
                
                try:
                    result6 = supabase.table('tracks').select('*').filter('description', 'ilike', f'%{tag}%').execute()
                    field_results.append(result6.data or [])
                except Exception as e:
                    log_debug(f"Description search failed for '{tag}': {e}")
                    field_results.append([])
                
                # 3. Licensing tier filters
                if tag.lower() in ['instant', 'standard', 'bespoke']:
                    log_debug(f"Searching for licensing tier: '{tag.lower()}'")
                    try:
                        licensing_result = supabase.table('tracks').select('*').filter('licensing_tier', 'eq', tag.lower()).execute()
                        log_debug(f"Licensing search for '{tag.lower()}' returned: {len(licensing_result.data) if licensing_result.data else 0} results")
                        field_results.append(licensing_result.data or [])
                    except Exception as licensing_error:
                        log_error(f"Licensing search failed for '{tag.lower()}'", licensing_error)
                        field_results.append([])
                
                # 4. Price range filters
                if tag == "Under $1K":
                    log_debug(f"Searching for price range: Under $1K")
                    try:
                        # Fixed price under 1000 OR max price under 1000
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'lt', 1000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('max_price', 'lt', 1000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        log_debug(f"Price search 'Under $1K' returned: {len(price_data)} results")
                        field_results.append(price_data)
                    except Exception as price_error:
                        log_error(f"Price search failed for 'Under $1K'", price_error)
                        field_results.append([])
                elif tag == "$1K-$5K":
                    log_debug(f"Searching for price range: $1K-$5K")
                    try:
                        # Fixed price between 1000-5000 OR min/max price in range
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'gte', 1000).filter('fixed_price', 'lte', 5000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('min_price', 'gte', 1000).filter('max_price', 'lte', 5000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        log_debug(f"Price search '$1K-$5K' returned: {len(price_data)} results")
                        field_results.append(price_data)
                    except Exception as price_error:
                        log_error(f"Price search failed for '$1K-$5K'", price_error)
                        field_results.append([])
                elif tag == "$5K-$25K":
                    try:
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'gte', 5000).filter('fixed_price', 'lte', 25000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('min_price', 'gte', 5000).filter('max_price', 'lte', 25000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        field_results.append(price_data)
                    except Exception:
                        field_results.append([])
                elif tag == "$25K-$50K":
                    try:
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'gte', 25000).filter('fixed_price', 'lte', 50000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('min_price', 'gte', 25000).filter('max_price', 'lte', 50000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        field_results.append(price_data)
                    except Exception:
                        field_results.append([])
                elif tag == "$50K-$100K":
                    try:
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'gte', 50000).filter('fixed_price', 'lte', 100000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('min_price', 'gte', 50000).filter('max_price', 'lte', 100000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        field_results.append(price_data)
                    except Exception:
                        field_results.append([])
                elif tag == "$100K+":
                    try:
                        # Fixed price over 100000 OR max price over 100000
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'gt', 100000).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('max_price', 'gt', 100000).execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        field_results.append(price_data)
                    except Exception:
                        field_results.append([])
                elif tag == "Free":
                    try:
                        # Fixed price is 0 OR no pricing set
                        price_result1 = supabase.table('tracks').select('*').filter('fixed_price', 'eq', 0).execute()
                        price_result2 = supabase.table('tracks').select('*').filter('fixed_price', 'is', 'null').filter('min_price', 'is', 'null').execute()
                        price_data = (price_result1.data or []) + (price_result2.data or [])
                        field_results.append(price_data)
                    except Exception:
                        field_results.append([])
                
                # For this tag, take UNION of all field searches (any field match counts)
                if field_results and any(result_list for result_list in field_results):
                    tag_tracks = {}
                    for result_list in field_results:
                        for track in result_list:
                            tag_tracks[track['id']] = track
                    
                    tag_track_list = list(tag_tracks.values())
                    
                    if tag_track_list:
                        all_tag_results.append(set(track['id'] for track in tag_track_list))
                        log_info(f"Found {len(tag_track_list)} tracks matching tag '{tag}' in any field")
                
            # Now combine results from all tags
            if all_tag_results:
                if len(all_tag_results) == 1:
                    # Single tag - use all results
                    final_track_ids = all_tag_results[0]
                    log_info(f"Single tag search: {len(final_track_ids)} tracks found")
                else:
                    # Multiple tags - use intersection (tracks must match ALL tags)
                    final_track_ids = all_tag_results[0]
                    for tag_result_set in all_tag_results[1:]:
                        final_track_ids = final_track_ids.intersection(tag_result_set)
                    log_info(f"Multi-tag search: {len(final_track_ids)} tracks match ALL {len(normalized_tags)} tags")
                
                # Get full track data for final IDs
                if final_track_ids:
                    all_tracks = supabase.table('tracks').select('*').in_('id', list(final_track_ids)).execute()
                    tag_results = all_tracks.data or []
                    tag_matched = True
                else:
                    tag_results = []
            else:
                tag_results = []
                
            if tag_matched and tag_results:
                # Remove duplicates by track ID  
                unique_tracks = {}
                for track in tag_results:
                    unique_tracks[track['id']] = track
                
                filtered_result = type('Result', (), {'data': list(unique_tracks.values())})()
                log_success(f"Tag filtering successful: {len(filtered_result.data)} unique tracks found")
            else:
                filtered_result = None
                log_warning("No tag matches found")
            
            # Now apply vector search to tag-filtered results
            if filtered_result and filtered_result.data:
                log_info("Applying vector similarity to tag-filtered tracks")
                
                # Calculate similarities manually for tag-filtered tracks
                tracks_with_similarity = []
                for track in filtered_result.data:
                    if track.get('embedding'):
                        try:
                            # Convert string embedding back to list
                            track_embedding = eval(track['embedding']) if isinstance(track['embedding'], str) else track['embedding']
                            
                            # Calculate cosine similarity (simplified)
                            from numpy import dot
                            from numpy.linalg import norm
                            similarity = dot(query_embedding, track_embedding) / (norm(query_embedding) * norm(track_embedding))
                            
                            if similarity > match_threshold:
                                track['similarity'] = float(similarity)
                                tracks_with_similarity.append(track)
                        except Exception as sim_error:
                            log_debug(f"Similarity calculation failed for track {track.get('id')}: {sim_error}")
                            # Include track anyway if it matched tags
                            track['similarity'] = 0.8  # Default similarity for tag matches
                            tracks_with_similarity.append(track)
                
                # Sort by similarity and limit results
                tracks_with_similarity.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                final_results = tracks_with_similarity[:match_count]
                
                if final_results:
                    log_step(5, 5, f"Returning {len(final_results)} tag-filtered results with similarity")
                    log_success(f"Search completed successfully with {len(final_results)} results")
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_description, "audio") if audio_description else None
                    video_summary = create_analysis_summary(video_description, "video") if video_description else None
                    
                    return {
                        "results": final_results,
                        "message": f"Found {len(final_results)} tracks matching tags and similarity",
                        "search_type": "tag_filtered_vector",
                        "applied_tags": normalized_tags,
                        "audioAnalysis": audio_description if audio_description else None,
                        "videoAnalysis": video_description if video_description else None,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
                else:
                    # Return tag-filtered results without vector similarity
                    log_warning("No vector similarity matches, returning tag-filtered results only")
                    log_step(5, 5, f"Returning {len(filtered_result.data[:match_count])} tag-only results")
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_description, "audio") if audio_description else None
                    video_summary = create_analysis_summary(video_description, "video") if video_description else None
                    
                    return {
                        "results": filtered_result.data[:match_count],
                        "message": f"Found {len(filtered_result.data[:match_count])} tracks matching tags",
                        "search_type": "tag_only",
                        "applied_tags": normalized_tags,
                        "audioAnalysis": audio_description if audio_description else None,
                        "videoAnalysis": video_description if video_description else None,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
            else:
                # No tag matches, fall back to vector search
                log_warning("No tracks match required tags, trying fallback search")
                
                fallback_threshold = match_threshold * 0.7
                log_info(f"Executing fallback vector search with lowered threshold: {fallback_threshold}")
                
                # Use manual vector search instead of broken function
                try:
                    all_tracks = supabase.table('tracks').select('*').filter('embedding', 'not.is', 'null').execute()
                    
                    if not all_tracks.data:
                        log_step(5, 5, "Fallback search completed - no tracks with embeddings")
                        # Create narrative summaries even when no tracks available
                        audio_narrative = audio_description if audio_description else None
                        video_narrative = video_description if video_description else None
                        # Create summaries for UI display
                        audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                        video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                        
                        return {
                            "results": [],
                            "message": "No tracks available for search",
                            "search_type": "no_tracks",
                            "audioAnalysis": audio_narrative,
                            "videoAnalysis": video_narrative,
                            "audioSummary": audio_summary,
                            "videoSummary": video_summary
                        }
                    
                    # Calculate similarities manually for fallback
                    fallback_results = []
                    for track in all_tracks.data:
                        try:
                            track_embedding = eval(track['embedding']) if isinstance(track['embedding'], str) else track['embedding']
                            
                            from numpy import dot
                            from numpy.linalg import norm
                            similarity = dot(query_embedding, track_embedding) / (norm(query_embedding) * norm(track_embedding))
                            
                            if similarity > fallback_threshold:
                                formatted_track = {
                                    'id': track['id'],
                                    'title': track.get('title', ''),
                                    'artist': track.get('artist', ''),
                                    'bpm': track.get('bpm'),
                                    'key': track.get('key'),
                                    'genre': ', '.join(track.get('genre', [])) if track.get('genre') else '',
                                    'mood': ', '.join(track.get('moods', {}).get('moods', [])) if track.get('moods') else '',
                                    'instruments': ', '.join(track.get('instruments', [])) if track.get('instruments') else '',
                                    'vocal_type': track.get('vocal_type'),
                                    'description': track.get('description', ''),
                                    'duration': track.get('duration'),
                                    'explicit_content': track.get('explicit_content', False),
                                    'language': track.get('language', ''),
                                    'audio_url': track.get('audio_url', ''),
                                    'similarity': float(similarity)
                                }
                                fallback_results.append(formatted_track)
                        except Exception as sim_error:
                            log_debug(f"Fallback similarity calculation failed for track {track.get('id')}: {sim_error}")
                            continue
                    
                    # Sort and limit fallback results
                    fallback_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    final_fallback = fallback_results[:match_count]
                    
                    log_step(5, 5, f"Fallback search completed - {len(final_fallback)} results")
                    
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_description, "audio") if audio_description else None
                    video_summary = create_analysis_summary(video_description, "video") if video_description else None
                    
                    return {
                        "results": final_fallback,
                        "message": f"No exact tag matches found. Showing {len(final_fallback)} similar results.",
                        "search_type": "fallback_vector",
                        "original_tags": normalized_tags,
                        "audioAnalysis": audio_description if audio_description else None,
                        "videoAnalysis": video_description if video_description else None,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
                    
                except Exception as fallback_error:
                    log_error("Fallback vector search failed", fallback_error)
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_description, "audio") if audio_description else None
                    video_summary = create_analysis_summary(video_description, "video") if video_description else None
                    
                    return {
                        "results": [],
                        "message": "Fallback search failed",
                        "search_type": "fallback_failed",
                        "original_tags": normalized_tags,
                        "error": str(fallback_error),
                        "audioAnalysis": audio_description if audio_description else None,
                        "videoAnalysis": video_description if video_description else None,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
            
        else:
            # No tags - pure vector search using match_embeddings database function
            log_info("No tags provided - executing pure vector search")
            
            try:
                # First try using the database function for optimal performance
                log_debug("Attempting to use match_embeddings database function")
                
                # Call the database function
                result = supabase.rpc('match_embeddings', {
                    'query_embedding': query_embedding,
                    'match_threshold': match_threshold,
                    'match_count': match_count
                }).execute()
                
                if result.data:
                    # Debug: Print what the database function returns
                    print(f"[DEBUG] Database function returned {len(result.data)} results")
                    for i, track in enumerate(result.data[:2]):  # Print first 2 tracks
                        print(f"[DEBUG] Track {i+1} keys: {list(track.keys())}")
                        print(f"[DEBUG] Track {i+1} audio_url: {track.get('audio_url', 'MISSING')}")
                    
                    # WORKAROUND: Get audio URLs separately since the function doesn't return them
                    track_ids = [track['id'] for track in result.data]
                    audio_url_lookup = {}
                    
                    if track_ids:
                        try:
                            audio_urls_result = supabase.table('tracks').select('id, audio_url').in_('id', track_ids).execute()
                            if audio_urls_result.data:
                                audio_url_lookup = {track['id']: track.get('audio_url', '') for track in audio_urls_result.data}
                                print(f"[DEBUG] Fetched audio URLs for {len(audio_url_lookup)} tracks")
                                print(f"[DEBUG] Sample audio URLs: {list(audio_url_lookup.values())[:2]}")
                        except Exception as e:
                            print(f"[WARNING] Failed to fetch audio URLs: {e}")
                    
                    # Format the results
                    formatted_results = []
                    for track in result.data:
                        # Get audio_url from lookup or fallback to track data
                        track_audio_url = audio_url_lookup.get(track.get('id'), track.get('audio_url', ''))
                        
                        formatted_track = {
                            'id': track.get('id'),
                            'title': track.get('title', ''),
                            'artist': track.get('artist', ''),
                            'bpm': track.get('bpm'),
                            'key': track.get('key'),
                            'genre': track.get('genre', ''),
                            'mood': track.get('mood', ''),
                            'instruments': track.get('instruments', ''),
                            'vocal_type': track.get('vocal_type'),
                            'description': track.get('description', ''),
                            'duration': track.get('duration'),
                            'explicit_content': track.get('explicit_content', False),
                            'language': track.get('language', ''),
                            'audio_url': track_audio_url,
                            'similarity': float(track.get('similarity', 0))
                        }
                        
                        # Generate AI reasoning for this match
                        try:
                            if combined_description:
                                formatted_track['match_reasoning'] = generate_match_reasoning(
                                    combined_description, 
                                    formatted_track, 
                                    formatted_track['similarity']
                                )
                            else:
                                formatted_track['match_reasoning'] = f"This track has a {formatted_track['similarity']:.1%} similarity match with your search."
                        except Exception as reasoning_error:
                            log_warning(f"Failed to generate reasoning for track {formatted_track['id']}: {reasoning_error}")
                            formatted_track['match_reasoning'] = f"Good match with {formatted_track['similarity']:.1%} similarity."
                        
                        formatted_results.append(formatted_track)
                    
                    log_step(5, 5, f"Database vector search completed - {len(formatted_results)} results")
                    log_success(f"Database vector search successful with {len(formatted_results)} results")
                    
                    # Create narrative summaries and store uploads
                    audio_narrative = None
                    video_narrative = None
                    
                    if audio and audio_description:
                        print("\nProcessing Audio...")
                        # Generate detailed analysis
                        try:
                            print("🎵 Analyzing audio content...")
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            analysis_prompt = f"""Analyze this audio file in detail and provide a comprehensive description including:

1. Musical Characteristics:
   - Genre and subgenre identification
   - Mood and emotional impact
   - Energy level and dynamics
   - Key musical elements (tempo, key, etc.)
   - Production quality and style

2. Use Case Analysis:
   - What types of projects/content would this track be ideal for?
   - Key emotional responses it might evoke
   - Target audience and setting recommendations

3. Matching Criteria:
   - What makes this track distinctive?
   - Key elements that make it suitable for specific uses
   - Standout characteristics for search matching

Please provide a detailed, professional analysis focusing on practical applications and search relevance.

Audio file: {audio.filename}
Current analysis: {audio_description}
"""
                            response = model.generate_content(analysis_prompt)
                            if response and response.text:
                                audio_narrative = response.text.strip()
                                print(f"Generated detailed audio analysis: {len(audio_narrative)} chars")
                            else:
                                audio_narrative = audio_description
                        except Exception as e:
                            print(f"Error generating detailed analysis: {str(e)}")
                            audio_narrative = audio_description
                        
                        print(f"Audio Narrative Length: {len(audio_narrative)}")
                        # Store in search_index table
                        await store_search_upload(
                            title=audio.filename,
                            description=audio_narrative,
                            embedding=query_embedding,
                            content_type="audio"
                        )
                    
                    if video and video_description:
                        video_narrative = video_description
                        # Store in search_index table
                        await store_search_upload(
                            title=video.filename,
                            description=video_narrative,
                            embedding=query_embedding,
                            content_type="video"
                        )
                    
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                    video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                    
                    return {
                        "results": formatted_results,
                        "message": f"Found {len(formatted_results)} similar tracks",
                        "search_type": "vector_only",
                        "audioAnalysis": audio_narrative,
                        "videoAnalysis": video_narrative,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
                else:
                    log_step(5, 5, "Database vector search completed - no results above threshold")
                    log_warning("No database vector search results found above similarity threshold")
                    
                    # Create narrative summaries and store uploads even if no results
                    audio_narrative = None
                    video_narrative = None
                    
                    if audio and audio_description:
                        audio_narrative = audio_description
                        await store_search_upload(
                            title=audio.filename,
                            description=audio_narrative,
                            embedding=query_embedding,
                            content_type="audio"
                        )
                    
                    if video and video_description:
                        video_narrative = video_description
                        await store_search_upload(
                            title=video.filename,
                            description=video_narrative,
                            embedding=query_embedding,
                            content_type="video"
                        )
                    
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                    video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                    
                    return {
                        "results": [],
                        "message": f"No tracks found with similarity above {match_threshold}",
                        "search_type": "no_results",
                        "audioAnalysis": audio_narrative,
                        "videoAnalysis": video_narrative,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
                    
            except Exception as db_error:
                log_warning(f"Database function failed, falling back to manual calculation: {db_error}")
                
                # Fallback to manual calculation only if database function fails
                try:
                    # Get all tracks with embeddings
                    all_tracks = supabase.table('tracks').select('*').filter('embedding', 'not.is', 'null').execute()
                    
                    if not all_tracks.data:
                        log_warning("No tracks with embeddings found")
                        # Create narrative summaries even when no tracks available
                        audio_narrative = audio_description if audio_description else None
                        video_narrative = video_description if video_description else None
                        # Create summaries for UI display
                        audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                        video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                        
                        return {
                            "results": [],
                            "message": "No tracks available for search",
                            "search_type": "no_tracks",
                            "audioAnalysis": audio_narrative,
                            "videoAnalysis": video_narrative,
                            "audioSummary": audio_summary,
                            "videoSummary": video_summary
                        }
                    
                    # Calculate similarities manually
                    tracks_with_similarity = []
                    for track in all_tracks.data:
                        try:
                            # Convert string embedding back to list if needed
                            track_embedding = eval(track['embedding']) if isinstance(track['embedding'], str) else track['embedding']
                            
                            # Calculate cosine similarity
                            from numpy import dot
                            from numpy.linalg import norm
                            similarity = dot(query_embedding, track_embedding) / (norm(query_embedding) * norm(track_embedding))
                            
                            if similarity > match_threshold:
                                # Format track data to match expected schema
                                formatted_track = {
                                    'id': track['id'],
                                    'title': track.get('title', ''),
                                    'artist': track.get('artist', ''),
                                    'bpm': track.get('bpm'),
                                    'key': track.get('key'),
                                    'genre': ', '.join(track.get('genre', [])) if track.get('genre') else '',
                                    'mood': ', '.join(track.get('moods', {}).get('moods', [])) if track.get('moods') else '',
                                    'instruments': ', '.join(track.get('instruments', [])) if track.get('instruments') else '',
                                    'vocal_type': track.get('vocal_type'),
                                    'description': track.get('description', ''),
                                    'duration': track.get('duration'),
                                    'explicit_content': track.get('explicit_content', False),
                                    'language': track.get('language', ''),
                                    'audio_url': track.get('audio_url', ''),
                                    'similarity': float(similarity)
                                }
                                
                                # Generate AI reasoning for this match (manual search fallback)
                                try:
                                    if combined_description:
                                        formatted_track['match_reasoning'] = generate_match_reasoning(
                                            combined_description, 
                                            formatted_track, 
                                            formatted_track['similarity']
                                        )
                                    else:
                                        formatted_track['match_reasoning'] = f"This track has a {formatted_track['similarity']:.1%} similarity match with your search."
                                except Exception as reasoning_error:
                                    log_warning(f"Failed to generate reasoning for track {formatted_track['id']}: {reasoning_error}")
                                    formatted_track['match_reasoning'] = f"Good match with {formatted_track['similarity']:.1%} similarity."
                                
                                tracks_with_similarity.append(formatted_track)
                        except Exception as sim_error:
                            log_debug(f"Similarity calculation failed for track {track.get('id')}: {sim_error}")
                            continue
                    
                    # Sort by similarity and limit results
                    tracks_with_similarity.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    final_results = tracks_with_similarity[:match_count]
                    
                    if final_results:
                        log_step(5, 5, f"Manual vector search completed - {len(final_results)} results")
                        log_success(f"Manual vector search successful with {len(final_results)} results")
                        
                        # Use raw analysis without narrative summary pressure
                        audio_narrative = None
                        video_narrative = None
                        
                        if audio and audio_description:
                            audio_narrative = audio_description  # Use raw analysis
                            await store_search_upload(
                                title=audio.filename,
                                description=audio_description,  # Store raw analysis
                                embedding=query_embedding,
                                content_type="audio"
                            )
                        
                        if video and video_description:
                            video_narrative = video_description  # Use raw analysis
                            await store_search_upload(
                                title=video.filename,
                                description=video_description,  # Store raw analysis
                                embedding=query_embedding,
                                content_type="video"
                            )
                        
                        # Create summaries for UI display
                        audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                        video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                        
                        return {
                            "results": final_results,
                            "message": f"Found {len(final_results)} similar tracks (manual calculation)",
                            "search_type": "vector_manual",
                            "audioAnalysis": audio_narrative,
                            "videoAnalysis": video_narrative,
                            "audioSummary": audio_summary,
                            "videoSummary": video_summary
                        }
                    else:
                        log_step(5, 5, "Manual vector search completed - no results above threshold")
                        log_warning("No manual vector search results found above similarity threshold")
                        
                        # Use raw analysis without narrative summary pressure
                        audio_narrative = None
                        video_narrative = None
                        
                        if audio and audio_description:
                            audio_narrative = audio_description  # Use raw analysis
                            await store_search_upload(
                                title=audio.filename,
                                description=audio_description,  # Store raw analysis
                                embedding=query_embedding,
                                content_type="audio"
                            )
                        
                        if video and video_description:
                            video_narrative = video_description  # Use raw analysis
                            await store_search_upload(
                                title=video.filename,
                                description=video_description,  # Store raw analysis
                                embedding=query_embedding,
                                content_type="video"
                            )
                        
                        # Create summaries for UI display
                        audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                        video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                        
                        return {
                            "results": [],
                            "message": f"No tracks found with similarity above {match_threshold}",
                            "search_type": "no_results",
                            "audioAnalysis": audio_narrative,
                            "videoAnalysis": video_narrative,
                            "audioSummary": audio_summary,
                            "videoSummary": video_summary
                        }
                        
                except Exception as search_error:
                    log_error("Manual vector search failed", search_error)
                    # Create narrative summaries even on error
                    audio_narrative = audio_description if audio_description else None
                    video_narrative = video_description if video_description else None
                    # Create summaries for UI display
                    audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
                    video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
                    
                    return {
                        "error": f"Vector search failed: {str(search_error)}", 
                        "audioAnalysis": audio_narrative, 
                        "videoAnalysis": video_narrative,
                        "audioSummary": audio_summary,
                        "videoSummary": video_summary
                    }
    
    except Exception as e:
        log_error("Search execution failed", e)
        # Create narrative summaries even on error
        audio_narrative = audio_description if audio_description else None
        video_narrative = video_description if video_description else None
        # Create summaries for UI display
        audio_summary = create_analysis_summary(audio_narrative, "audio") if audio_narrative else None
        video_summary = create_analysis_summary(video_narrative, "video") if video_narrative else None
        
        return {
            "error": str(e), 
            "audioAnalysis": audio_narrative, 
            "videoAnalysis": video_narrative,
            "audioSummary": audio_summary,
            "videoSummary": video_summary
        }

@app.post("/fix-database-function")
async def fix_database_function():
    """Temporary endpoint to fix the match_embeddings function"""
    
    try:
        # Use existing Supabase client connection
        # Create a simple test query first to check if our current function is working
        test_embedding = [0.1] * 768
        
        # Test the current function
        current_result = supabase.rpc('match_embeddings', {
            'query_embedding': test_embedding,
            'match_threshold': 0.01,
            'match_count': 1
        }).execute()
        
        print("[DEBUG] Current function result keys:", list(current_result.data[0].keys()) if current_result.data else "No results")
        
        # Since we can't directly execute CREATE OR REPLACE with Supabase client,
        # let's modify the backend to handle audio_url properly instead
        return {
            "success": False, 
            "message": "Direct SQL execution not available through Supabase client. Need to use psql or different approach.",
            "current_function_keys": list(current_result.data[0].keys()) if current_result.data else []
        }
        
    except Exception as e:
        print(f"[ERROR] Error testing function: {e}")
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {
        "message": "Zen Music Analyzer API is running!",
        "status": "healthy",
        "version": "v10-ultimate"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with authentication and model testing"""
    status = {
        "status": "unknown",
        "timestamp": datetime.now().isoformat(),
        "authentication": {
            "method": auth_method,
            "google_ai_studio": "not_tested",
            "vertex_ai": "not_tested",
            "models_available": 0
        },
        "database": {
            "supabase": "not_tested"
        },
        "environment": {
            "google_project_id": os.getenv("GOOGLE_PROJECT_ID"),
            "has_api_key": bool(os.getenv("GOOGLE_AI_API_KEY")),
            "has_service_account": bool(os.getenv("GEMINI_SERVICE_ACCOUNT_KEY"))
        }
    }
    
    # Test Google AI Studio
    try:
        models = list(genai.list_models())
        gemini_models = [m for m in models if 'gemini-2.5-flash' in m.name]
        status["authentication"]["google_ai_studio"] = "[SUCCESS] working"
        status["authentication"]["models_available"] = len(models)
        status["authentication"]["gemini_flash_variants"] = len(gemini_models)
    except Exception as e:
        status["authentication"]["google_ai_studio"] = f"[ERROR] failed: {str(e)}"
    
    # Test Supabase
    try:
        response = supabase.table('tracks').select('id').limit(1).execute()
        status["database"]["supabase"] = "[SUCCESS] working"
    except Exception as e:
        status["database"]["supabase"] = f"[ERROR] failed: {str(e)}"
    
    # Determine overall status
    if (status["authentication"]["google_ai_studio"].startswith("[SUCCESS]") and
        status["database"]["supabase"].startswith("[SUCCESS]")):
        status["status"] = "[SUCCESS] healthy"
    else:
        status["status"] = "[WARNING] degraded"
    
    return status

@app.get("/test-model")
async def test_model():
    """Test Gemini model with a simple query"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Test message")
        
        return {
            "status": "healthy",
            "model": "gemini-2.5-flash",
            "response_length": len(str(response.text)) if response.text else 0,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "[ERROR] failed",
            "error": str(e),
            "auth_method": auth_method
        }

def create_analysis_summary(analysis_text: str, content_type: str = "audio") -> str:
    """Create a concise, engaging summary of analysis text using Gemini 2.5 Flash"""
    if not analysis_text:
        print("No analysis text provided")
        return None
    
    try:
        print("Generating comprehensive detailed summary...")
        if content_type == "audio":
            summary_prompt = f"""Create a comprehensive, detailed summary of this audio analysis. Provide an in-depth analysis that thoroughly explains why this track is valuable and how it matches search criteria.

**Analysis Text:**
{analysis_text}

**Required Elements - Cover Each In Detail:**
1. Main characteristics:
   - Genre analysis and sub-genres
   - Detailed mood and emotional qualities
   - Production style and techniques
   - Sonic characteristics

2. Key matching features and unique qualities:
   - Distinctive sonic elements
   - Production techniques
   - Arrangement structure
   - Musical complexity

3. Technical analysis:
   - Instrumentation breakdown
   - Mix quality and characteristics
   - Sound design elements
   - Production techniques used

4. Commercial and practical aspects:
   - Target audience
   - Ideal use cases
   - Licensing potential
   - Similar artists/tracks

5. Matching criteria analysis:
   - Why it matches the search
   - Similarity factors
   - Unique selling points
   - Differentiation factors

Provide a detailed analysis between 500-1000 characters (strictly enforced). Focus on the most relevant aspects while maintaining depth and clarity. Prioritize quality insights over exhaustive length.

Summary:"""
        else:
            summary_prompt = f"""Create a comprehensive, detailed summary of this video analysis. Provide an in-depth analysis focusing on visual style, technical qualities, and content compatibility.

**Analysis Text:**
{analysis_text}

**Required Elements - Cover Each In Detail:**
1. Visual Analysis:
   - Cinematography style and techniques
   - Color grading and visual treatment
   - Lighting characteristics
   - Shot composition and framing

2. Technical Qualities:
   - Production value assessment
   - Post-production techniques
   - Special effects or motion graphics
   - Sound design and audio elements

3. Content Analysis:
   - Narrative structure
   - Pacing and rhythm
   - Visual storytelling elements
   - Emotional impact

4. Compatibility Factors:
   - Target audience alignment
   - Brand compatibility
   - Usage scenarios
   - Platform suitability

5. Distinctive Features:
   - Unique visual elements
   - Creative approaches
   - Technical innovations
   - Stand-out qualities

Provide a detailed analysis (500-1000 characters) that thoroughly examines all aspects. Include specific examples and technical observations where relevant.

Summary:"""
        
        # Generate summary using Gemini
        print("Using Gemini for detailed summary...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(summary_prompt)
        
        if not response or not response.text:
            print("Failed to generate summary")
            return None
            
        summary = response.text.strip()
        print(f"Raw summary length: {len(summary)} characters")
        
        # Clean up the response
        summary = re.sub(r'^["\']|["\']$', '', summary)  # Remove quotes
        summary = re.sub(r'^\s*Summary:\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^\s*Your summary:\s*', '', summary, flags=re.IGNORECASE)
        summary = summary.strip()
        
        # Strict length control
        def trim_to_length(text: str, max_length: int) -> str:
            """Trim text to max length at sentence boundary"""
            if len(text) <= max_length:
                return text
                
            # Find last complete sentence before limit
            breakpoint = text[:max_length].rfind('.')
            if breakpoint == -1:
                # If no sentence break found, look for other punctuation
                for punct in ['!', '?', ';']:
                    breakpoint = text[:max_length].rfind(punct)
                    if breakpoint != -1:
                        break
                if breakpoint == -1:
                    # If still no break found, look for paragraph
                    breakpoint = text[:max_length].rfind('\n')
                    if breakpoint == -1:
                        # Last resort: break at word boundary
                        breakpoint = text[:max_length].rfind(' ')
                        if breakpoint == -1:
                            breakpoint = max_length - 1
            
            return text[:breakpoint + 1].strip()

        # Apply length constraints
        if len(summary) < 500:
            print(f"⚠️ Summary too short ({len(summary)} chars). Regenerating with more detail...")
            return create_analysis_summary(analysis_text, content_type)  # Retry
        elif len(summary) > 1000:
            print(f"⚠️ Summary too long ({len(summary)} chars). Trimming to 1000...")
            summary = trim_to_length(summary, 1000)
            print(f"✂️ Trimmed summary length: {len(summary)} chars")
        
    except Exception as e:
        print(f" Error generating summary: {str(e)}")
        return None
    
    try:
        log_info(f"Generating {content_type} summary for text of length {len(analysis_text)}")
        
        # Create different prompts based on content type
        if content_type == "audio":
            summary_prompt = f"""Create a concise, engaging 1-2 sentence summary of this music analysis that describes what the song sounds like and feels like. Focus on genre, mood, energy, and the overall listening experience.

**Full Analysis:**
{analysis_text}

**Requirements:**
- Maximum 150 characters
- Focus on mood, genre, and emotional feeling
- Use descriptive, engaging language
- Make it sound appealing and relatable
- Don't use technical jargon

**Example:** "An atmospheric progressive house track with ethereal vocals and building energy that creates a dreamy, introspective mood perfect for late-night listening."

**Your summary:**"""

        else:  # video
            summary_prompt = f"""Create a concise, engaging 1-2 sentence summary of this video analysis that describes what the video shows and its atmosphere. Focus on the visual style, mood, and overall impression.

**Full Analysis:**
{analysis_text}

**Requirements:**
- Maximum 150 characters
- Focus on visual style, atmosphere, and content
- Use descriptive, engaging language
- Make it appealing and relatable
- Don't use technical jargon

**Example:** "A dynamic basketball commercial showcasing athletic prowess in minimalist studio settings with energetic music and modern visual aesthetics."

**Your summary:**"""

        # Generate summary using Gemini 2.5 Flash
        if auth_method == "google_ai_studio":
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(summary_prompt)
            summary = response.text.strip()
            
            # Clean up the response
            summary = re.sub(r'^["\']|["\']$', '', summary)  # Remove quotes
            summary = re.sub(r'^\s*Your summary:\s*', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'^\s*Summary:\s*', '', summary, flags=re.IGNORECASE)
            summary = summary.strip()
            
            # Ensure it's not too long
            if len(summary) > 180:
                # Try to cut at sentence boundary
                sentences = summary.split('. ')
                summary = sentences[0]
                if not summary.endswith('.'):
                    summary += '.'
            
            log_debug(f"Generated {content_type} summary ({len(summary)} chars): {summary}")
            return summary
            
        else:
            # Fallback to template-based summary
            return create_template_summary(analysis_text, content_type)
            
    except Exception as e:
        log_warning(f"Gemini summary generation failed for {content_type}: {e}")
        # Fall back to template-based summary
        return create_template_summary(analysis_text, content_type)

def create_template_summary(analysis_text: str, content_type: str = "audio") -> str:
    """Create template-based summary as fallback"""
    if not analysis_text:
        return ""
    
    # Extract key information for different content types
    if content_type == "audio":
        # Look for genre, mood, and style information
        lines = analysis_text.split('\n')
        key_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['genre', 'style', 'mood', 'tempo', 'energy', 'instruments']):
                # Clean up formatting
                clean_line = re.sub(r'^[\*\-\•]\s*', '', line)
                clean_line = re.sub(r'^\*\*[^*]+\*\*:?\s*', '', clean_line)
                if clean_line and len(clean_line) > 10:
                    key_info.append(clean_line)
                    if len(key_info) >= 2:  # Limit to 2 key points for brevity
                        break
        
        if key_info:
            summary = " • ".join(key_info)
            if len(summary) > 150:
                summary = key_info[0]  # Just take the first one if too long
            return summary
    
    elif content_type == "video":
        # Look for visual and thematic information
        lines = analysis_text.split('\n')
        key_info = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['shows', 'features', 'depicts', 'style', 'mood', 'atmosphere', 'visual']):
                clean_line = re.sub(r'^[\*\-\•]\s*', '', line)
                clean_line = re.sub(r'^\*\*[^*]+\*\*:?\s*', '', clean_line)
                if clean_line and len(clean_line) > 10:
                    key_info.append(clean_line)
                    if len(key_info) >= 2:
                        break
        
        if key_info:
            summary = " • ".join(key_info)
            if len(summary) > 150:
                summary = key_info[0]
            return summary
    
    # Fallback: return first sentence(s) up to ~150 chars
    sentences = re.split(r'[.!?]+', analysis_text)
    summary = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(summary + sentence) <= 130:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip()

def generate_match_reasoning(search_description: str, track_data: dict, similarity_score: float) -> str:
    """Generate AI reasoning for why a track matches the search using Gemini 2.5 Flash"""
    try:
        log_debug(f"Generating match reasoning for track: {track_data.get('title', 'Unknown')}")
        
        # Prepare track information for analysis
        track_title = track_data.get('title', 'Unknown')
        track_artist = track_data.get('artist', 'Unknown Artist')
        track_genre = track_data.get('genre', 'Unknown')
        track_mood = track_data.get('mood', 'Not specified')
        track_instruments = track_data.get('instruments', 'Not specified')
        track_bpm = track_data.get('bpm', 'Unknown')
        track_key = track_data.get('key', 'Unknown')
        track_vocals = track_data.get('vocal_type', 'Not specified')
        track_description = track_data.get('description', 'No description available')

        # Create an engaging reasoning prompt
        reasoning_prompt = f"""You are an AI music curator explaining why a track is perfect for someone's search. Write an engaging, specific explanation that highlights the key matching elements.

**User's Search Context:**
{search_description[:1000]}...

**Matched Track Details:**
• Title: "{track_title}" by {track_artist}
• Genre: {track_genre}
• Mood/Energy: {track_mood}
• Instruments: {track_instruments}
• BPM: {track_bpm} | Key: {track_key}
• Vocals: {track_vocals}
• Description: {track_description[:300]}...
• Match Strength: {similarity_score:.1%}

**Instructions:**
- Write exactly 1-2 sentences (maximum 200 characters)
- Be specific about WHY it matches (don't just say "it matches")
- Focus on the most compelling connections (genre, mood, energy, style, instruments)
- Use engaging, enthusiastic language that makes the track sound appealing
- Mention similarity strength only if notably high (>70%)
- Make it sound like a personal recommendation from a music expert

**Examples:**
• "This progressive house track perfectly captures the ethereal, building energy of your audio with its atmospheric synths and emotional vocal delivery."
• "The trap-influenced beats and modern production style align beautifully with the urban energy and dynamic pace of your video content."
• "This ambient electronic piece matches your introspective mood perfectly, featuring the same dreamy textures and contemplative atmosphere."

**Your compelling match explanation:**"""

        # Generate reasoning using Gemini 2.5 Flash
        try:
            if auth_method == "google_ai_studio":
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(reasoning_prompt)
                reasoning = response.text.strip()
                
                # Clean up the response
                reasoning = re.sub(r'^["\']|["\']$', '', reasoning)  # Remove quotes
                reasoning = re.sub(r'^\s*Your.*explanation:\s*', '', reasoning, flags=re.IGNORECASE)
                reasoning = re.sub(r'^\s*Match explanation:\s*', '', reasoning, flags=re.IGNORECASE)
                reasoning = re.sub(r'^\s*Explanation:\s*', '', reasoning, flags=re.IGNORECASE)
                reasoning = reasoning.strip()
                
                # Ensure it's concise
                if len(reasoning) > 250:
                    sentences = reasoning.split('. ')
                    reasoning = sentences[0]
                    if not reasoning.endswith('.'):
                        reasoning += '.'
                
                # Validate minimum quality
                if len(reasoning) < 20 or 'match' not in reasoning.lower():
                    log_warning("Generated reasoning too short or generic, using template")
                    return generate_template_reasoning(search_description, track_data, similarity_score)
                
                log_debug(f"Generated match reasoning ({len(reasoning)} chars): {reasoning[:100]}...")
                return reasoning
                
            else:
                # Fallback to template-based reasoning
                return generate_template_reasoning(search_description, track_data, similarity_score)
                
        except Exception as gemini_error:
            log_warning(f"Gemini reasoning generation failed: {gemini_error}")
            # Fall back to template-based reasoning
            return generate_template_reasoning(search_description, track_data, similarity_score)
            
    except Exception as e:
        log_error(f"Match reasoning generation failed: {e}")
        return f"This track has a {similarity_score:.1%} similarity match with your search criteria."

def generate_template_reasoning(search_description: str, track_data: dict, similarity_score: float) -> str:
    """Generate template-based reasoning as fallback"""
    try:
        title = track_data.get('title', 'track')
        genre = track_data.get('genre', '').lower()
        mood = track_data.get('mood', '').lower()
        instruments = track_data.get('instruments', '').lower()
        
        # Analyze search description for keywords
        search_lower = search_description.lower()
        
        match_points = []
        
        # Genre matching
        if genre:
            if 'trap' in genre and any(word in search_lower for word in ['urban', 'street', 'modern', 'hip-hop']):
                match_points.append("trap/hip-hop style aligns with the urban aesthetic")
            elif 'electronic' in genre and any(word in search_lower for word in ['modern', 'digital', 'futuristic']):
                match_points.append("electronic elements match the modern production style")
            elif genre:
                match_points.append(f"{genre} genre suits the content style")
        
        # Mood matching
        if mood:
            if 'energetic' in mood or 'high-energy' in mood:
                if any(word in search_lower for word in ['dynamic', 'active', 'movement', 'action']):
                    match_points.append("high-energy mood complements the dynamic visuals")
            elif 'ambient' in mood or 'calm' in mood:
                if any(word in search_lower for word in ['peaceful', 'serene', 'ambient']):
                    match_points.append("ambient atmosphere matches the peaceful tone")
        
        # Similarity score context
        if similarity_score > 0.7:
            strength = "strong"
        elif similarity_score > 0.5:
            strength = "good"
        else:
            strength = "moderate"
        
        # Construct reasoning
        if match_points:
            reasoning = f"This track shows a {strength} match where the {', '.join(match_points[:2])}."
        else:
            reasoning = f"This track has a {strength} similarity match ({similarity_score:.1%}) with your search criteria based on musical and stylistic elements."
        
        # Add similarity context if notably high
        if similarity_score > 0.7:
            reasoning += f" The {similarity_score:.1%} similarity score indicates particularly strong alignment."
        
        return reasoning
        
    except Exception as e:
        log_error(f"Template reasoning generation failed: {e}")
        return f"This track matches your search with {similarity_score:.1%} similarity."

@app.post("/api/analyze-instant")
async def instant_analysis(
    audio: UploadFile = File(None),
    video: UploadFile = File(None),
    search_mode: str = Form("deep")  # "bolt" or "deep"
):
    """Instant analysis endpoint for immediate file processing on upload - supports parallel processing"""
    log_separator("INSTANT ANALYSIS")
    log_info(f"Instant analysis request:")
    log_info(f"  - Audio: {audio.filename if audio else None}")
    log_info(f"  - Video: {video.filename if video else None}")
    log_info(f"  - Mode: {search_mode.upper()}")
    
    # Choose analysis function based on search mode
    analysis_function = analyze_content_with_gemini_bolt if search_mode.lower() == "bolt" else analyze_content_with_gemini
    
    try:
        # OPTIMIZED: Process both files in parallel using asyncio.gather
        import asyncio
        
        async def process_audio():
            if audio:
                log_info(f"Processing audio file: {audio.filename} ({search_mode.upper()} mode)")
                audio_bytes = await audio.read()
                result = analysis_function(audio.filename, audio_bytes, "audio")
                if result:
                    log_success(f"Instant {search_mode.upper()} audio analysis completed: {len(result)} chars")
                return result
            return None
        
        async def process_video():
            if video:
                log_info(f"Processing video file: {video.filename} ({search_mode.upper()} mode)")
                video_bytes = await video.read()
                result = analysis_function(video.filename, video_bytes, "video")
                if result:
                    log_success(f"Instant {search_mode.upper()} video analysis completed: {len(result)} chars")
                return result
            return None
        
        # Run both analyses in parallel
        audio_description, video_description = await asyncio.gather(
            process_audio(),
            process_video(),
            return_exceptions=True
        )
        
        # Handle any exceptions from parallel processing
        if isinstance(audio_description, Exception):
            log_warning(f"Audio processing failed: {audio_description}")
            audio_description = None
        if isinstance(video_description, Exception):
            log_warning(f"Video processing failed: {video_description}")
            video_description = None
        
        # Return results for both files
        results = {}
        
        if audio and audio_description:
            results["audio"] = {
                "success": True,
                "type": "audio",
                "filename": audio.filename,
                "analysis": audio_description,
                "mode": search_mode,
                "character_count": len(audio_description)
            }
        elif audio and not audio_description:
            results["audio"] = {
                "success": False,
                "type": "audio", 
                "filename": audio.filename,
                "error": "Audio analysis failed"
            }
            
        if video and video_description:
            results["video"] = {
                "success": True,
                "type": "video",
                "filename": video.filename,
                "analysis": video_description,
                "mode": search_mode,
                "character_count": len(video_description)
            }
        elif video and not video_description:
            results["video"] = {
                "success": False,
                "type": "video",
                "filename": video.filename, 
                "error": "Video analysis failed"
            }
        
        if not results:
            log_warning("No files provided for instant analysis")
            return {"error": "No files provided"}
        
        # For backward compatibility, if only one file was processed, return the single result
        if len(results) == 1:
            return list(results.values())[0]
        
        # For multiple files, return both results
        return {
            "success": True,
            "results": results,
            "mode": search_mode,
            "parallel_processing": True
        }
            
    except Exception as e:
        log_error("Instant analysis failed", e)
        return {"error": str(e)}

# Start the server
if __name__ == "__main__":
    import uvicorn
    log_info("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 