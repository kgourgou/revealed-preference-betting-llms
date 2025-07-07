import pytest
from src.games import simple_problem, prime_problem
from src.utils import prime_counting_function


class TestSimpleProblem:
    """Test suite for the simple_problem function."""

    def test_simple_problem_default_parameters(self):
        """Test simple_problem with default parameters."""
        result = simple_problem()

        assert isinstance(result, dict)
        assert "prompt" in result
        assert "true_prob_a" in result
        assert "true_prob_b" in result
        assert "payout_a" in result
        assert "payout_b" in result

        assert result["true_prob_a"] == 0.1
        assert result["true_prob_b"] == 0.9
        assert result["payout_a"] is None
        assert result["payout_b"] is None

        # Check that prompt contains expected content
        prompt = result["prompt"]
        assert "10-sided die" in prompt
        assert "Bet A" in prompt
        assert "Bet B" in prompt
        assert "maximize your expected winnings" in prompt

    def test_simple_problem_with_payouts(self):
        """Test simple_problem with specific payout values."""
        result = simple_problem(payout_a=10.0, payout_b=5.0)

        assert result["payout_a"] == 10.0
        assert result["payout_b"] == 5.0
        assert result["true_prob_a"] == 0.1
        assert result["true_prob_b"] == 0.9

        # Check that prompt contains the payout values
        prompt = result["prompt"]
        assert "$10.0" in prompt
        assert "$5.0" in prompt

    def test_simple_problem_probability_calculation(self):
        """Test that probabilities are calculated correctly."""
        result = simple_problem()

        # Bet A: win if die shows 1 (1 outcome out of 10)
        assert result["true_prob_a"] == 0.1

        # Bet B: win if die shows 2-10 (9 outcomes out of 10)
        assert result["true_prob_b"] == 0.9

        # Probabilities should sum to 1
        prob_sum = result["true_prob_a"] + result["true_prob_b"]
        assert prob_sum == pytest.approx(1.0)

    def test_simple_problem_prompt_structure(self):
        """Test that the prompt has the expected structure."""
        result = simple_problem(payout_a=15.0, payout_b=3.0)
        prompt = result["prompt"]

        # Check for key elements in the prompt
        assert "perfectly rational gambler" in prompt
        assert "maximize your expected winnings" in prompt
        assert "10-sided die" in prompt
        assert "fair" in prompt
        assert "Bet A:" in prompt
        assert "Bet B:" in prompt
        assert "die 1" in prompt  # Bet A condition
        assert "die shows (2, 3, 4, 5, 6, 7, 8, 9, or 10)" in prompt  # Bet B condition
        assert "$15.0" in prompt
        assert "$3.0" in prompt
        assert "Pick which bet" in prompt
        assert "A or B" in prompt


class TestPrimeProblem:
    """Test suite for the prime_problem function."""

    def test_prime_problem_default_parameters(self):
        """Test prime_problem with default parameters."""
        result = prime_problem(prime_bound=10)

        assert isinstance(result, dict)
        assert "prompt" in result
        assert "true_prob_a" in result
        assert "true_prob_b" in result
        assert "payout_a" in result
        assert "payout_b" in result

        assert result["payout_a"] is None
        assert result["payout_b"] is None

        # Check that prompt contains expected content
        prompt = result["prompt"]
        assert "10 sides" in prompt
        assert "prime number" in prompt
        assert "non-prime number" in prompt
        assert "Answer: A" in prompt
        assert "Answer: B" in prompt

    def test_prime_problem_with_payouts(self):
        """Test prime_problem with specific payout values."""
        result = prime_problem(prime_bound=20, payout_a=25.0, payout_b=8.0)

        assert result["payout_a"] == 25.0
        assert result["payout_b"] == 8.0

        # Check that prompt contains the payout values
        prompt = result["prompt"]
        assert "$25.0" in prompt
        assert "$8.0" in prompt

    def test_prime_problem_probability_calculation(self):
        """Test that probabilities are calculated correctly for different bounds."""
        # Test with prime_bound = 10
        result_10 = prime_problem(prime_bound=10)
        primes_under_10 = prime_counting_function(10)  # Should be 4 (2, 3, 5, 7)
        expected_prob_a_10 = primes_under_10 / 10  # 4/10 = 0.4
        expected_prob_b_10 = 1 - expected_prob_a_10  # 6/10 = 0.6

        assert result_10["true_prob_a"] == pytest.approx(expected_prob_a_10)
        assert result_10["true_prob_b"] == pytest.approx(expected_prob_b_10)

        # Test with prime_bound = 20
        result_20 = prime_problem(prime_bound=20)
        primes_under_20 = prime_counting_function(20)  # Should be 8
        expected_prob_a_20 = primes_under_20 / 20  # 8/20 = 0.4
        expected_prob_b_20 = 1 - expected_prob_a_20  # 12/20 = 0.6

        assert result_20["true_prob_a"] == pytest.approx(expected_prob_a_20)
        assert result_20["true_prob_b"] == pytest.approx(expected_prob_b_20)

    def test_prime_problem_probability_sum(self):
        """Test that probabilities always sum to 1."""
        test_bounds = [5, 10, 15, 20, 25, 30]

        for bound in test_bounds:
            result = prime_problem(prime_bound=bound)
            prob_sum = result["true_prob_a"] + result["true_prob_b"]
            assert prob_sum == pytest.approx(1.0)

    def test_prime_problem_prompt_structure(self):
        """Test that the prompt has the expected structure."""
        result = prime_problem(prime_bound=15, payout_a=20.0, payout_b=10.0)
        prompt = result["prompt"]

        # Check for key elements in the prompt
        assert "perfectly rational gambler" in prompt
        assert "maximize your expected winnings" in prompt
        assert "15 sides" in prompt
        assert "fair" in prompt
        assert "prime number" in prompt
        assert "non-prime number" in prompt
        assert "including 1" in prompt
        assert "DO NOT output OR generate ANY reasoning" in prompt
        assert "Answer: A" in prompt
        assert "Answer: B" in prompt
        assert "$20.0" in prompt
        assert "$10.0" in prompt

    def test_prime_problem_edge_cases(self):
        """Test prime_problem with edge case values."""
        # Test with very small bound
        result_small = prime_problem(prime_bound=2)
        # Only 2 is prime out of 2 numbers (1, 2)
        assert result_small["true_prob_a"] == pytest.approx(0.5)
        # 1 is non-prime out of 2 numbers
        assert result_small["true_prob_b"] == pytest.approx(0.5)

        # Test with bound = 3
        result_3 = prime_problem(prime_bound=3)
        assert result_3["true_prob_a"] == pytest.approx(2 / 3)  # 2 and 3 are prime
        assert result_3["true_prob_b"] == pytest.approx(1 / 3)  # Only 1 is non-prime

    def test_prime_problem_large_bound(self):
        """Test prime_problem with larger bounds."""
        result_100 = prime_problem(prime_bound=100)
        primes_under_100 = prime_counting_function(100)  # Should be 25
        expected_prob_a = primes_under_100 / 100  # 25/100 = 0.25
        expected_prob_b = 1 - expected_prob_a  # 75/100 = 0.75

        assert result_100["true_prob_a"] == pytest.approx(expected_prob_a)
        assert result_100["true_prob_b"] == pytest.approx(expected_prob_b)

    def test_prime_problem_template_substitution(self):
        """Test that template substitution works correctly."""
        result = prime_problem(prime_bound=12, payout_a=50.0, payout_b=5.0)
        prompt = result["prompt"]

        # Check that template variables are properly substituted
        assert "12 sides" in prompt
        assert "$50.0" in prompt
        assert "$5.0" in prompt

        # Check that the bound is used in probability calculation
        primes_under_12 = prime_counting_function(12)  # Should be 5 (2, 3, 5, 7, 11)
        expected_prob_a = primes_under_12 / 12  # 5/12
        expected_prob_b = 1 - expected_prob_a  # 7/12

        assert result["true_prob_a"] == pytest.approx(expected_prob_a)
        assert result["true_prob_b"] == pytest.approx(expected_prob_b)


class TestIntegration:
    """Integration tests between games and utils modules."""

    def test_prime_problem_uses_prime_counting_function(self):
        """Test that prime_problem correctly uses prime_counting_function."""
        # Mock prime_counting_function to verify it's called
        with pytest.MonkeyPatch().context() as m:
            m.setattr("src.games.prime_counting_function", lambda x: 42)

            result = prime_problem(prime_bound=100)
            assert result["true_prob_a"] == pytest.approx(42 / 100)
            assert result["true_prob_b"] == pytest.approx(58 / 100)

    def test_probability_consistency(self):
        """Test that probabilities are consistent across different problem types."""
        # Simple problem: fixed probabilities
        simple_result = simple_problem()
        assert simple_result["true_prob_a"] == 0.1
        assert simple_result["true_prob_b"] == 0.9

        # Prime problem: calculated probabilities
        prime_result = prime_problem(prime_bound=10)
        # For bound=10, primes are 2,3,5,7 (4 primes)
        assert prime_result["true_prob_a"] == 0.4
        assert prime_result["true_prob_b"] == 0.6

        # Both should have probabilities that sum to 1
        simple_sum = simple_result["true_prob_a"] + simple_result["true_prob_b"]
        assert simple_sum == pytest.approx(1.0)
        prime_sum = prime_result["true_prob_a"] + prime_result["true_prob_b"]
        assert prime_sum == pytest.approx(1.0)
