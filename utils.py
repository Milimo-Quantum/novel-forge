import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_style_prompt(style_file_path: str = "prompts/write_essay.md") -> str:
    """
    Load the Paul Graham style instructions from the markdown file.
    Extracts the IDENTITY and OUTPUT INSTRUCTIONS sections only.
    """
    try:
        with open(style_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Extract from '# IDENTITY and PURPOSE' to 'EXAMPLE PAUL GRAHAM ESSAYS'
        identity_start = content.find("# IDENTITY and PURPOSE")
        example_start = content.find("EXAMPLE PAUL GRAHAM ESSAYS")
        identity_section = content[identity_start:example_start].strip() if identity_start != -1 and example_start != -1 else ""

        # Extract from last '# OUTPUT INSTRUCTIONS' to end
        last_output_idx = content.rfind("# OUTPUT INSTRUCTIONS")
        output_section = content[last_output_idx:].strip() if last_output_idx != -1 else ""

        combined = f"{identity_section}\n\n{output_section}"
        return combined.strip()
    except Exception as e:
        logger.error(f"Failed to load style prompt: {e}")
        return ""

def log_operation(func):
    """Decorator to log operation metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting operation: {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Operation {func.__name__} failed: {str(e)}")
            raise
    return wrapper

async def get_ollama_usage_stats(model_name: str) -> Dict[str, Any]:
    """Get usage statistics with enhanced monitoring"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:11434/api/show', params={'name': model_name})
            response.raise_for_status()
            stats = response.json()
            
            # Add monitoring data
            stats['last_accessed'] = datetime.now().isoformat()
            stats['api_call_count'] = stats.get('api_call_count', 0) + 1
            return stats
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        return {}

async def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed model info with monitoring"""
    try:
        stats = await get_ollama_usage_stats(model_name)
        if stats:
            return {
                'name': model_name,
                'parameters': stats.get('parameters', {}),
                'template': stats.get('template', ''),
                'system': stats.get('system', ''),
                'license': stats.get('license', 'Unknown'),
                'monitoring': {
                    'last_accessed': stats.get('last_accessed'),
                    'api_calls': stats.get('api_call_count', 0)
                }
            }
        return None
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return None

async def is_model_ready(model_name: str) -> bool:
    """Check model readiness with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'http://localhost:11434/api/generate',
                    json={'model': model_name, 'prompt': ''}
                )
                if response.status_code == 200:
                    return True
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                    
                return False
        except Exception as e:
            logger.warning(f"Model check attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            return False

def get_model_tags() -> List[Dict[str, Any]]:
    """Get model tags with enhanced error handling"""
    try:
        return asyncio.run(_get_model_tags_async())
    except Exception as e:
        logger.error(f"Error getting model tags: {str(e)}")
        return []

async def _get_model_tags_async() -> List[Dict[str, Any]]:
    """Async implementation with monitoring"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:11434/api/tags')
            response.raise_for_status()
            models = response.json().get('models', [])
            
            # Add monitoring data
            for model in models:
                model['last_checked'] = datetime.now().isoformat()
            return models
    except Exception as e:
        logger.error(f"Error getting model tags: {str(e)}")
        return []

def format_model_details(model: Dict[str, Any]) -> str:
    """Format model details with monitoring info"""
    try:
        details = [
            f"Model: {model.get('name', 'Unknown')}",
            f"Size: {model.get('size', 'Unknown')}",
            f"Modified: {model.get('modified', 'Unknown')}",
            f"Digest: {model.get('digest', 'Unknown')[:8]}...",
            f"Last Checked: {model.get('last_checked', 'Never')}",
            f"API Calls: {model.get('api_call_count', 0)}"
        ]
        return "\n".join(details)
    except Exception as e:
        logger.error(f"Error formatting model details: {str(e)}")
        return "Error getting model details"

async def cleanup_model_resources(model_name: str) -> bool:
    """Clean up resources with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    'http://localhost:11434/api/delete',
                    json={'name': model_name}
                )
                if response.status_code == 200:
                    logger.info(f"Successfully cleaned up model {model_name}")
                    return True
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                    
                logger.warning(f"Failed to clean up model {model_name}")
                return False
        except Exception as e:
            logger.warning(f"Cleanup attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            return False

def cleanup_model_resources_sync(model_name: str) -> bool:
    """Synchronous wrapper with monitoring"""
    try:
        return asyncio.run(cleanup_model_resources(model_name))
    except Exception as e:
        logger.error(f"Error in synchronous cleanup: {str(e)}")
        return False

def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config with enhanced monitoring"""
    try:
        normalized = {
            'base_url': config.get('base_url', 'http://localhost:11434'),
            'model': config.get('model', ''),
            'temperature': float(config.get('temperature', 0.7)),
            'monitoring': {
                'config_validated': datetime.now().isoformat()
            }
        }

        if not normalized['model']:
            raise ValueError("Model name is required")

        return normalized
    except Exception as e:
        logger.error(f"Error validating model config: {str(e)}")
        raise

async def get_system_resources() -> Dict[str, Any]:
    """Get system resource usage"""
    try:
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'timestamp': datetime.now().isoformat()
        }
    except ImportError:
        logger.warning("psutil not available, skipping resource monitoring")
        return {}
    except Exception as e:
        logger.error(f"Error getting system resources: {str(e)}")
        return {}

def validate_concept_integrity(initial: str, refined: str) -> bool:
    """
    Checks if the refined concept is semantically related to the initial concept.
    Simple substring check; can be replaced with NLP/fuzzy logic if needed.
    """
    if not initial or not refined:
        return True  # Don't block if one is missing; logging will catch this
    return initial.lower() in refined.lower() or refined.lower() in initial.lower()
