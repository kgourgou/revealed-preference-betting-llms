import pytest
import os
from unittest.mock import Mock, patch
from src.utils import (
    prime_counting_function,
    load_client,
    generate_content,
    TEMPERATURE,
)


class TestPrimeCountingFunction:
    """Test suite for the prime_counting_function."""

    def test_prime_counting_function_basic(self):
        """Test basic prime counting functionality."""
        assert prime_counting_function(10) == 4  # primes: 2, 3, 5, 7
        # primes: 2, 3, 5, 7, 11, 13, 17, 19
        assert prime_counting_function(20) == 8
        # primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
        assert prime_counting_function(30) == 10

    def test_prime_counting_function_edge_cases(self):
        """Test edge cases for prime counting function."""
        assert prime_counting_function(1) == 0  # No primes <= 1
        assert prime_counting_function(2) == 1  # Only 2 is prime
        assert prime_counting_function(3) == 2  # 2 and 3 are prime
        assert prime_counting_function(0) == 0  # No primes <= 0

    def test_prime_counting_function_large_numbers(self):
        """Test with larger numbers to ensure correctness."""
        assert prime_counting_function(50) == 15
        assert prime_counting_function(100) == 25
        assert prime_counting_function(1000) == 168

    def test_prime_counting_function_caching(self):
        """Test that the function uses caching correctly."""
        # Clear the cache first
        prime_counting_function.cache_clear()

        # First call should compute
        result1 = prime_counting_function(10)

        # Second call should use cache
        result2 = prime_counting_function(10)

        assert result1 == result2 == 4

        # Verify cache info shows hits
        cache_info = prime_counting_function.cache_info()
        assert cache_info.hits >= 1

    def test_prime_counting_function_known_primes(self):
        """Test with known prime numbers to verify accuracy."""
        # Test first few prime numbers
        primes_under_50 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        assert prime_counting_function(50) == len(primes_under_50)

        # Test specific ranges
        # primes: 2, 3, 5, 7, 11, 13, 17, 19, 23
        assert prime_counting_function(25) == 9


class TestLoadClient:
    """Test suite for the load_client function."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
    @patch("src.utils.OpenAI")
    def test_load_client_success(self, mock_openai):
        """Test successful client creation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = load_client("test_model")

        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test_api_key"
        )
        assert client == mock_client

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.utils.OpenAI")
    def test_load_client_no_api_key(self, mock_openai):
        """Test client creation when API key is not set."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = load_client("test_model")

        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key=None
        )
        assert client == mock_client


class TestGenerateContent:
    """Test suite for the generate_content function."""

    @patch("src.utils.load_client")
    def test_generate_content_standard_model(self, mock_load_client):
        """Test content generation with standard model."""
        mock_client = Mock()
        mock_load_client.return_value = mock_client

        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated content"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        result = generate_content("test prompt", "gpt-4")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=TEMPERATURE,
        )
        assert result == "Generated content"

    @patch("src.utils.load_client")
    def test_generate_content_gpt_oss_model(self, mock_load_client):
        """Test content generation with GPT-OSS model (includes system message)."""
        mock_client = Mock()
        mock_load_client.return_value = mock_client

        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Generated content"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        result = generate_content("test prompt", "gpt-oss-model")

        expected_messages = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": "test prompt"},
        ]

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-oss-model", messages=expected_messages, temperature=TEMPERATURE
        )
        assert result == "Generated content"

    @patch("src.utils.load_client")
    def test_generate_content_liquid_ai_model(self, mock_load_client):
        """Test content generation with LiquidAI model (returns None)."""
        mock_client = Mock()
        mock_load_client.return_value = mock_client

        # LiquidAI models don't use the standard flow, they fall through
        result = generate_content("test prompt", "LiquidAI/LFM2-700M")

        # Should return None since no return statement is reached
        assert result is None

        # Should not call any client methods
        mock_client.chat.completions.create.assert_not_called()
        mock_client.models.generate_content.assert_not_called()

    @patch("src.utils.load_client")
    def test_generate_content_empty_response(self, mock_load_client):
        """Test handling of empty response."""
        mock_client = Mock()
        mock_load_client.return_value = mock_client

        mock_completion = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = ""
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        result = generate_content("test prompt", "gpt-4")

        assert result == ""

    @patch("src.utils.load_client")
    def test_generate_content_no_choices(self, mock_load_client):
        """Test handling of response with no choices."""
        mock_client = Mock()
        mock_load_client.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = []
        mock_client.chat.completions.create.return_value = mock_completion

        with pytest.raises(IndexError):
            generate_content("test prompt", "gpt-4")


class TestConstants:
    """Test suite for constants."""

    def test_temperature_constant(self):
        """Test that TEMPERATURE constant is set correctly."""
        assert TEMPERATURE == 0.0
        assert isinstance(TEMPERATURE, float)
