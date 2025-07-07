"""
Pytest configuration and common fixtures for the test suite.
"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()

    # Mock chat completion response
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Mock response"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_completion

    # Mock models.generate_content for Gemini
    mock_gemini_response = Mock()
    mock_gemini_response.text = "Mock Gemini response"
    client.models.generate_content.return_value = mock_gemini_response

    return client


@pytest.fixture
def sample_prime_numbers():
    """Sample prime numbers for testing."""
    return {
        10: [2, 3, 5, 7],  # 4 primes
        20: [2, 3, 5, 7, 11, 13, 17, 19],  # 8 primes
        30: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],  # 10 primes
        50: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],  # 15 primes
        100: 25,  # Count of primes under 100
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    return {"OPENROUTER_API_KEY": "test_api_key_12345"}


@pytest.fixture(autouse=True)
def clear_prime_counting_cache():
    """Clear the prime_counting_function cache before each test."""
    from src.utils import prime_counting_function

    prime_counting_function.cache_clear()
    yield
    prime_counting_function.cache_clear()
