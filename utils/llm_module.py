"""LLM interaction utility module.

This module provides functions for interacting with external LLM services
for generating text, processing inputs, and structured outputs.
"""

import json
import logging
import random
import time
from typing import Any

import requests

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised for errors in the LLM module."""


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    structured_output: bool = False,
    retry_count: int = 3,
    retry_delay: float = 2.0,
) -> str | dict[str, Any] | None:
    """Generate a completion using the configured LLM service.

    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt content
        temperature: Temperature setting (creativity)
        max_tokens: Maximum tokens to generate
        structured_output: Whether to return structured JSON
        retry_count: Number of retries on failure
        retry_delay: Delay between retries

    Returns:
        Generated text or structured data, or None if generation fails

    """
    # Use configured temperature if not specified
    if temperature is None:
        temperature = config.model.temperature

    # Use configured max_tokens if not specified
    if max_tokens is None or max_tokens < 0:
        max_tokens = config.model.max_tokens if config.model.max_tokens > 0 else None

    # Set up the API request
    headers = {
        "Content-Type": "application/json",
    }

    # Add structured output instructions if needed
    if structured_output:
        system_prompt += (
            "\n\nProvide your response in JSON format with the following structure:\n"
        )
        system_prompt += (
            '{\n  "understanding": "your interpretation of the situation",\n'
        )
        system_prompt += '  "response": "your nurturing response to the child",\n'
        system_prompt += '  "action": "specific action you\'re taking (comfort, teach, play, etc.)"\n}'

    # Construct the API request body
    payload = {
        "model": config.model.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    # Add max_tokens if specified
    if max_tokens:
        payload["max_tokens"] = max_tokens

    # Attempt the API request with retries
    for attempt in range(retry_count):
        try:
            # Make the API request
            response = requests.post(
                config.server.llm_server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )

            # Check for successful response
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]

                # Parse JSON for structured output
                if structured_output:
                    try:
                        # Extract JSON content (handle cases where it might be wrapped in ```json blocks)
                        if "```json" in content:
                            # Extract content between ```json and ```
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            if json_end == -1:  # If no closing ``` found
                                json_end = len(content)
                            json_content = content[json_start:json_end].strip()
                        elif "```" in content:
                            # Extract content between ``` and ```
                            json_start = content.find("```") + 3
                            json_end = content.find("```", json_start)
                            if json_end == -1:  # If no closing ``` found
                                json_end = len(content)
                            json_content = content[json_start:json_end].strip()
                        else:
                            # Assume the entire content is JSON
                            json_content = content

                        # Parse the JSON
                        result = json.loads(json_content)
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from LLM response: {e!s}")
                        logger.debug(f"Response content: {content}")
                        # Continue with retry if we couldn't parse the JSON
                        if attempt < retry_count - 1:
                            time.sleep(retry_delay)
                            continue
                        # Fallback: return raw text if all retries fail to parse JSON
                        return content
                else:
                    # Return raw text
                    return content
            else:
                logger.error(f"No choices in LLM response: {response_data}")

        except requests.exceptions.RequestException as e:
            logger.error(
                f"LLM API request failed (attempt {attempt + 1}/{retry_count}): {e!s}"
            )

        except Exception as e:
            logger.error(
                f"Unexpected error in LLM request (attempt {attempt + 1}/{retry_count}): {e!s}"
            )

        # Delay before retry, with jitter to avoid thundering herd
        if attempt < retry_count - 1:
            jitter = random.uniform(0.5, 1.5)
            time.sleep(retry_delay * jitter)

    # If we get here, all retries failed
    logger.error(f"All {retry_count} attempts to call LLM API failed")

    # Return None on failure
    return None


def get_embeddings(
    texts: list[str],
    retry_count: int = 3,
    retry_delay: float = 2.0,
) -> list[list[float]] | None:
    """Generate embeddings for the given texts.

    Args:
        texts: List of texts to embed
        retry_count: Number of retries on failure
        retry_delay: Delay between retries

    Returns:
        List of embedding vectors or None if embedding fails

    """
    # Set up the API request
    headers = {
        "Content-Type": "application/json",
    }

    # Construct the API request body
    payload = {
        "model": config.model.embedding_model,
        "input": texts,
    }

    # Attempt the API request with retries
    for attempt in range(retry_count):
        try:
            # Make the API request
            response = requests.post(
                config.server.embedding_server_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )

            # Check for successful response
            response.raise_for_status()
            response_data = response.json()

            if "data" in response_data and len(response_data["data"]) > 0:
                # Extract embeddings
                embeddings = [item["embedding"] for item in response_data["data"]]
                return embeddings
            logger.error(f"No data in embedding response: {response_data}")

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Embedding API request failed (attempt {attempt + 1}/{retry_count}): {e!s}"
            )

        except Exception as e:
            logger.error(
                f"Unexpected error in embedding request (attempt {attempt + 1}/{retry_count}): {e!s}"
            )

        # Delay before retry, with jitter to avoid thundering herd
        if attempt < retry_count - 1:
            jitter = random.uniform(0.5, 1.5)
            time.sleep(retry_delay * jitter)

    # If we get here, all retries failed
    logger.error(f"All {retry_count} attempts to call embedding API failed")

    # Return None on failure
    return None


def simulate_llm_response(
    system_prompt: str,
    user_prompt: str,
    structured_output: bool = False,
) -> str | dict[str, Any] | None:
    """Simulate an LLM response for testing/offline use.

    This function should only be used when the real LLM service is unavailable
    and is intended as a fallback for development and testing.

    Args:
        system_prompt: System prompt for the LLM
        user_prompt: User prompt content
        structured_output: Whether to return structured JSON

    Returns:
        Simulated response text or structured data

    """
    logger.warning("Using simulated LLM response - only for development/testing!")

    # Determine whether this is a mother response based on keywords
    is_mother = (
        "mother" in system_prompt.lower() or "nurturing" in system_prompt.lower()
    )

    if structured_output and is_mother:
        # Generate a simulated structured mother response
        return {
            "understanding": "Child appears to need attention and engagement.",
            "response": "I see you're exploring your world! That's wonderful!",
            "action": "encouragement",
        }
    if is_mother:
        # Generate a simulated unstructured mother response
        return "I see you're curious about that! Would you like to explore it together?"
    # Generic simulated response
    return "This is a simulated response for testing purposes."
