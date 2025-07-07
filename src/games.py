from outlines import Template
from src.utils import prime_counting_function


def simple_problem(
    payout_a: float | None = None, payout_b: float | None = None
) -> dict:
    template = Template.from_string(
        """
    You are a perfectly rational gambler and your sole objective is to maximize your expected winnings. You will be presented with two distinct bets based on the outcome of a single roll of a fair 10-sided die.

    Here are the bets:

    Bet A: You win if the die 1.
    If you win this bet, your payout is ${{payout_a}}.

    Bet B: You win if the die shows (2, 3, 4, 5, 6, 7, 8, 9, or 10) 
    If you win this bet, your payout is ${{payout_b}}.

    Pick which bet you would prefer to take, A or B, if you want to maximize your expected winnings. 
    """
    )

    return {
        "prompt": template(payout_a=payout_a, payout_b=payout_b),
        "true_prob_a": 0.1,
        "true_prob_b": 0.9,
        "payout_a": payout_a,
        "payout_b": payout_b,
    }


def prime_problem(
    prime_bound: int, payout_a: float | None = None, payout_b: float | None = None
) -> dict:
    template = Template.from_string(
        """
        You are a perfectly rational gambler and your sole objective is to maximize your expected winnings. The two bets are distinct and independent. The die has {{prime_bound}} sides and is fair.

        Pick which bet you would prefer to take, A or B, if you want to maximize your expected winnings. 

        Here are the bets:

        Bet A: You win if the die lands on a prime number.
        If you win this bet, your payout is ${{payout_a}}.

        Bet B: You win if the die shows a non-prime number (including 1).
        If you win this bet, your payout is ${{payout_b}}.

        DO NOT output OR generate ANY reasoning, just the letter of the bet you choose. Return with the format: 'Answer: A' or 'Answer: B'. Do not output anything else, just 'Answer: <letter>'.
        """
    )

    num_primes = prime_counting_function(prime_bound)
    return {
        "prompt": template(
            prime_bound=prime_bound, payout_a=payout_a, payout_b=payout_b
        ),
        "true_prob_a": num_primes / prime_bound,
        "true_prob_b": 1 - num_primes / prime_bound,
        "payout_a": payout_a,
        "payout_b": payout_b,
    }
