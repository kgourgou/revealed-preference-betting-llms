import argparse
from src.games import simple_problem, prime_problem
from src.utils import generate_content, TEMPERATURE
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple


from dotenv import load_dotenv

assert load_dotenv(".env")

PRIME_BOUND = 1024
DEFAULT_A_PAYOUT = 50.0


ids_to_openrouter_model_name = {
    "claude4": "anthropic/claude-sonnet-4",
    "claude37": "anthropic/claude-3.7-sonnet",
    "gpt4o": "openai/gpt-4o",
    "gpt41": "openai/gpt-4.1",
    "mistral": "mistralai/mistral-large",
    "hor-a": "openrouter/horizon-alpha",
    "hor-b": "openrouter/horizon-beta",
    "kimi-k2": "moonshotai/kimi-k2",
    "gpt20b": "openai/gpt-oss-20b",
    "lfm-7b": "liquid/lfm-7b",
    "lfm-3b": "liquid/lfm-3b",
}


class ExperimentConfig:
    """Centralized configuration for different experiment types."""

    def __init__(self, experiment_type: str, prime_bound: int = PRIME_BOUND):
        if experiment_type not in ["simple", "prime", "prime-inc"]:
            raise ValueError(f"Invalid experiment: {experiment_type}")
        self.experiment_type = experiment_type
        self.prime_bound = prime_bound

    def get_prompt(self, payout_b: float = None) -> dict:
        """Get the appropriate prompt based on experiment type."""
        if self.experiment_type == "simple":
            return simple_problem(payout_a=DEFAULT_A_PAYOUT, payout_b=payout_b)
        elif self.experiment_type == "prime":
            return prime_problem(
                prime_bound=self.prime_bound,
                payout_a=DEFAULT_A_PAYOUT,
                payout_b=payout_b,
            )
        elif self.experiment_type == "prime-inc":
            return prime_problem(
                prime_bound=self.prime_bound, payout_a=10, payout_b=payout_b
            )

    def get_base_prompt(self) -> dict:
        """Get the base prompt without payout_b parameter."""
        if self.experiment_type == "simple":
            return simple_problem()
        elif self.experiment_type == "prime":
            return prime_problem(prime_bound=self.prime_bound)
        elif self.experiment_type == "prime-inc":
            return prime_problem(prime_bound=self.prime_bound)


def query_llm_betting_choice(payout_b, model_name, experiment_config: ExperimentConfig):
    prompt = experiment_config.get_prompt(payout_b=payout_b)

    try:
        response = generate_content(
            prompt["prompt"],
            model_name=model_name,
        )
        llm_choice = response.strip().upper()
        print(f"LLM response: {llm_choice}")

        # parse the answer
        if "answer:" in llm_choice.lower():
            llm_choice = llm_choice.lower().split("answer:")[1].strip()
            llm_choice = llm_choice[0].upper()  # Get the first character (A or B)
            if llm_choice not in ["A", "B"]:
                print("Warning: LLM response does not contain a valid choice (A or B).")
                exit(1)
        else:
            print("Warning: LLM response does not contain 'Answer:' format.")
            exit(1)

        print(f"  Query (Payout B = ${payout_b:.8f}): LLM chose '{llm_choice}'")
        p_a, _ = estimate_probabilities_given_payout(payout_b, experiment_config)
        print(f"  Estimated P(A) at this payout: {p_a:.8f}")

        return llm_choice
    except Exception as e:
        print(f"Error during API call: {e}")
        return "error"


def find_indifference_payout(
    policy_func,
    num_iterations=15,
    model_name="gemini-2.5-flash",
    experiment_config: ExperimentConfig = None,
):
    """
    Uses a binary search to efficiently find the payout for Bet A where the LLM
    is indifferent between Bet A and Bet B.
    """

    prompt = experiment_config.get_base_prompt()

    TRUE_PROB_A = prompt["true_prob_a"]
    TRUE_PROB_B = prompt["true_prob_b"]
    FIXED_PAYOUT_A = prompt["payout_a"] or DEFAULT_A_PAYOUT
    print("Searching for the indifference payout for Bet B...")
    print(f"payout should be : {TRUE_PROB_A / TRUE_PROB_B * FIXED_PAYOUT_A:.2f}")
    low = 0
    high = 100

    prev_payout = 1
    for _ in range(num_iterations):
        payout_b = (low + high) / 2
        if abs(payout_b - prev_payout) / prev_payout < 1e-4:
            # If the change is very small, we can stop early
            print(f"Converged to payout: ${payout_b:.4f}")
            break

        choice = policy_func(
            payout_b=payout_b,
            model_name=model_name,
            experiment_config=experiment_config,
        )
        if choice == "error":
            return None

        if choice == "B":
            # LLM prefers Bet B. The tipping point must be at this payout or lower.
            high = payout_b
        else:  # 'A'
            # LLM prefers Bet A. The tipping point must be higher.
            low = payout_b

        prev_payout = payout_b

    return (low + high) / 2


def estimate_probabilities_given_payout(
    tipping_point_payout: float,
    experiment_config: ExperimentConfig,
) -> tuple[float, float]:
    prompt = experiment_config.get_base_prompt()

    FIXED_PAYOUT_A = prompt["payout_a"] or DEFAULT_A_PAYOUT
    # --- Calculate the Extracted Probabilities ---
    # At the tipping point, ExpectedValue(A) = ExpectedValue(B)
    # P_A * Payout_A = P_B * Payout_B

    # This can be rearranged to: P_A / P_B = Payout_B / Payout_A
    # Let R be the ratio of the implied probabilities, P_A / P_B
    implied_ratio_P_A_to_P_B = tipping_point_payout / FIXED_PAYOUT_A

    # We know P_A + P_B = 1  and  P_A = R * P_B
    # So, (R * P_B) + P_B = 1  -->  P_B * (R + 1) = 1
    # Which gives: P_B = 1 / (R + 1)
    implied_prob_b = 1 / (1 + implied_ratio_P_A_to_P_B)
    implied_prob_a = 1 - implied_prob_b
    return implied_prob_a, implied_prob_b


def create_incremental_prompt(
    base_prompt: str, cot_sentences: List[str], num_sentences: int
) -> str:
    """Create a prompt with the first num_sentences from COT reasoning."""
    if num_sentences == 0:
        return base_prompt

    reasoning_text = " ".join(cot_sentences[:num_sentences])

    base_prompt = base_prompt.replace(
        "DO NOT output OR generate ANY reasoning, just the letter of the bet you choose. "
        "Return with the format: 'Answer: A' or 'Answer: B'. "
        "Do not output anything else, just 'Answer: <letter>'.",
        "",
    )

    # Create the incremental prompt
    incremental_prompt = f"""{base_prompt}

    Here is some reasoning to help you think through this problem:

    {reasoning_text}

    Based on this reasoning, which bet would you prefer?  
    DO NOT output OR generate ANY reasoning, just the letter of the bet you choose. 
    Return with the format: 'Answer: A' or 'Answer: B'. 
    Do not output anything else, just 'Answer: <letter>'.
"""

    return incremental_prompt


def query_llm_incremental_choice(
    payout_b: float,
    model_name: str,
    experiment_config: ExperimentConfig,
    base_prompt: str,
    cot_sentences: List[str],
    num_sentences: int,
) -> str:
    """Query LLM with incremental COT reasoning."""

    prompt = create_incremental_prompt(base_prompt, cot_sentences, num_sentences)

    try:
        response = generate_content(prompt, model_name=model_name)
        llm_choice = response.strip().upper()

        # Parse the answer
        if "answer:" in llm_choice.lower():
            llm_choice = llm_choice.lower().split("answer:")[1].strip()
            llm_choice = llm_choice[0].upper()  # Get the first character (A or B)
            if llm_choice not in ["A", "B"]:
                print(
                    f"Warning: LLM response does not contain a valid choice "
                    f"(A or B): {llm_choice}"
                )
                return "error"
        else:
            print(
                f"Warning: LLM response does not contain 'Answer:' format: {llm_choice}"
            )
            return "error"

        return llm_choice
    except Exception as e:
        print(f"Error during API call: {e}")
        return "error"


def find_incremental_tipping_points(
    model_name: str, experiment_config: ExperimentConfig, num_iterations: int = 5
) -> List[Tuple[int, float]]:
    """Find tipping points for each incremental step of COT reasoning."""

    # Generate full COT reasoning
    print("Generating chain-of-thought reasoning...")

    def return_cot_sentences(p_a: float, p_b: float, payout_a: float, payout_b: float):
        diff_of_payouts = p_a * payout_a - p_b * payout_b
        result = "A" if diff_of_payouts > 0 else "B"

        return [
            f"bet A is p={p_a}",
            f"bet B is 1-p={p_b}",
            f"E_A=p * {payout_a} = {p_a * payout_a}",
            f"E_B = (1-p)* {payout_b} = {p_b * payout_b}",
            f"E_A-E_B = {diff_of_payouts}, so Bet {result} is better.",
        ]

    # Get the probabilities and payouts from the experiment configuration
    base_prompt_data = experiment_config.get_base_prompt()
    true_prob_a = base_prompt_data["true_prob_a"]
    true_prob_b = base_prompt_data["true_prob_b"]
    payout_a = base_prompt_data["payout_a"] or DEFAULT_A_PAYOUT
    # payout_b = 50.0  # Default payout for B for comparison

    tipping_points = []
    cot_sentences = return_cot_sentences(true_prob_a, true_prob_b, payout_a, 0)
    len_cot_sentences = len(cot_sentences) + 1

    # TODO debug this, for some reason it is not working correctly.
    for i in range(len_cot_sentences):  # +1 to include the case with no reasoning
        print(f"\n--- Step {i}/7 ---")
        print(f"Using first {i} parts of COT reasoning")

        # Create a function that uses the current number of sentences
        def query_with_sentences(payout_b, model_name, experiment_config):
            cot_sentences = return_cot_sentences(
                true_prob_a, true_prob_b, payout_a, payout_b
            )

            base_prompt = experiment_config.get_prompt(payout_b=payout_b)["prompt"]
            return query_llm_incremental_choice(
                payout_b, model_name, experiment_config, base_prompt, cot_sentences, i
            )

        # Find tipping point for this step
        tipping_point = find_indifference_payout(
            query_with_sentences,
            num_iterations=num_iterations,
            model_name=model_name,
            experiment_config=experiment_config,
        )

        if tipping_point is not None:
            tipping_points.append((i, tipping_point))
            print(f"Tipping point for step {i}: ${tipping_point:.4f}")
        else:
            print(f"Failed to find tipping point for step {i}")

    return tipping_points


def plot_incremental_results(
    tipping_points: List[Tuple[int, float]],
    true_tipping_point: float,
    experiment_name: str,
    model_name: str,
):
    """Plot the incremental tipping points."""
    if not tipping_points:
        print("No tipping points to plot")
        return

    steps, payouts = zip(*tipping_points)

    plt.figure(figsize=(12, 8))

    # Plot tipping points
    plt.plot(
        steps,
        payouts,
        "bo-",
        linewidth=2,
        markersize=8,
        label="Incremental Tipping Points",
    )

    # Plot true tipping point as horizontal line
    plt.axhline(
        y=true_tipping_point,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"True Tipping Point (${true_tipping_point:.4f})",
    )

    # Plot convergence line (last few points)
    if len(tipping_points) > 3:
        last_steps = steps[-3:]
        last_payouts = payouts[-3:]
        plt.plot(
            last_steps,
            last_payouts,
            "go-",
            linewidth=3,
            markersize=10,
            label="Convergence (Last 3 points)",
        )

    plt.xlabel("Number of COT Sentences Used")
    plt.ylabel("Tipping Point Payout ($)")
    plt.title(f"Incremental Tipping Points: {experiment_name}\nModel: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    plot_filename = f"incremental_tipping_points_{experiment_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {plot_filename}")

    # Log to mlflow
    mlflow.log_artifact(plot_filename)

    plt.show()


def create_incremental_table(
    tipping_points: List[Tuple[int, float]], true_tipping_point: float, model_name: str
) -> pd.DataFrame:
    """Create a table of incremental results."""
    if not tipping_points:
        return pd.DataFrame()

    data = []
    for step, payout in tipping_points:
        error = abs(payout - true_tipping_point)
        data.append(
            {
                "Model": model_name,
                "Step": step,
                "Tipping_Point": f"${payout:.4f}",
                "Error": f"${error:.4f}",
                "Error_Percentage": f"{error / true_tipping_point * 100:.2f}%",
            }
        )

    df = pd.DataFrame(data)
    return df


def main(
    experiment_name: str,
    model_name: str,
    num_iterations: int = 10,
    experiment: str = "prime",
    thinking: str = "False",
    prime_bound: int = PRIME_BOUND,
):
    print("Bet estimation experiment!")

    model_name = ids_to_openrouter_model_name.get(model_name, model_name)

    # Create experiment configuration
    experiment_config = ExperimentConfig(experiment, prime_bound=prime_bound)
    prompt = experiment_config.get_base_prompt()

    print("Loaded prompt")
    TRUE_PROB_A = prompt["true_prob_a"]
    TRUE_PROB_B = prompt["true_prob_b"]
    FIXED_PAYOUT_A = prompt["payout_a"] or DEFAULT_A_PAYOUT

    print("\n--- Betting Game Simulation ---")
    print(f"Using model: {model_name}")
    print(f"True Probabilities:      P(A)={TRUE_PROB_A:.4f}, P(B)={TRUE_PROB_B:.4f}")
    print(f"Fixed Payout for Bet A:  ${FIXED_PAYOUT_A:.2f}")

    # set mlflow to track the experiment in the local directory
    mlflow.set_tracking_uri("file:./mlruns")  # Use a local directory for tracking

    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(
        {
            "prompt": prompt,
            "true_prob_a": TRUE_PROB_A,
            "true_prob_b": TRUE_PROB_B,
            "fixed_payout_a": FIXED_PAYOUT_A,
            "model_name": model_name,
            "num_iterations": num_iterations,
            "experiment": experiment,
            "thinking": thinking,
            "temperature": TEMPERATURE,
            "prime_bound": prime_bound,
        }
    )

    if experiment == "prime-inc":
        # Run incremental experiment
        print("\n--- Running Incremental Experiment ---")

        # Calculate true tipping point for reference
        true_tipping_point = TRUE_PROB_A / TRUE_PROB_B * FIXED_PAYOUT_A

        # Find incremental tipping points
        tipping_points = find_incremental_tipping_points(
            model_name=model_name,
            experiment_config=experiment_config,
            num_iterations=num_iterations,
        )

        if tipping_points:
            print(f"\nFound {len(tipping_points)} incremental tipping points")

            # Create and display table
            table_df = create_incremental_table(
                tipping_points, true_tipping_point, model_name
            )
            print("\n--- Incremental Results Table ---")
            print(table_df.to_string(index=False))

            # Save table to CSV
            table_filename = f"incremental_results_{experiment_name}.csv"
            table_df.to_csv(table_filename, index=False)
            print(f"Table saved as {table_filename}")
            mlflow.log_artifact(table_filename)

            # Create and save plot
            plot_incremental_results(
                tipping_points, true_tipping_point, experiment_name, model_name
            )

            # Log all tipping points to mlflow
            for step, payout in tipping_points:
                mlflow.log_metric(f"tipping_point_step_{step}", payout)

            # Log final tipping point (full COT)
            final_tipping_point = tipping_points[-1][1] if tipping_points else None
            if final_tipping_point:
                mlflow.log_metric("final_tipping_point", final_tipping_point)
                mlflow.log_metric(
                    "convergence_error", abs(final_tipping_point - true_tipping_point)
                )
        else:
            print("Failed to find incremental tipping points")
    else:
        # Run standard experiment
        tipping_point_payout = find_indifference_payout(
            query_llm_betting_choice,
            num_iterations=num_iterations,
            model_name=model_name,
            experiment_config=experiment_config,
        )

        if tipping_point_payout is not None:
            print(
                f"\nDiscovered indifference payout for Bet B: ${tipping_point_payout:.2f}"
            )

            implied_prob_a, implied_prob_b = estimate_probabilities_given_payout(
                tipping_point_payout=tipping_point_payout,
                experiment_config=experiment_config,
            )

            print("\n--- Final Results ---")
            print(
                f"True Probabilities:      P(A)={TRUE_PROB_A:.8f}, P(B)={TRUE_PROB_B:.8f}"
            )
            print(
                f"Extracted Probabilities: P(A)={implied_prob_a:.8f}, P(B)={implied_prob_b:.8f}"
            )

            mlflow.log_metrics(
                {
                    "tipping_point_payout": tipping_point_payout,
                    "implied_prob_a": implied_prob_a,
                    "implied_prob_b": implied_prob_b,
                }
            )
        else:
            print(
                "Failed to find a valid tipping point payout. Check the logs for errors."
            )

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bet estimation experiment")
    parser.add_argument(
        "--mlflow_exp",
        type=str,
        default="test",
        help="Name of the MLflow experiment for tracking runs and metrics",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt4o",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for binary search (default: 10)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="prime",
        choices=["simple", "prime", "prime-inc"],
        help="Type of experiment (default: prime)",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="False",
        help="Thinking parameter (default: False)",
    )
    parser.add_argument(
        "--prime_bound",
        type=int,
        default=PRIME_BOUND,
        help=f"Prime bound for prime experiments (default: {PRIME_BOUND})",
    )

    args = parser.parse_args()

    main(
        experiment_name=args.mlflow_exp,
        model_name=args.model_name,
        num_iterations=args.num_iterations,
        experiment=args.experiment,
        thinking=args.thinking,
        prime_bound=args.prime_bound,
    )
