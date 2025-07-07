import os
from openai import OpenAI
from functools import lru_cache


TEMPERATURE = 0.0


# prime-counting function
@lru_cache(maxsize=None)
def prime_counting_function(n):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = []
    for i in range(2, n + 1):
        if is_prime(i):
            primes.append(i)
    return len(primes)


def load_client(model_name):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return client


def generate_content(prompt, model_name: str):
    client = load_client(model_name)

    if "gemini" in model_name:
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                thinking_config=ThinkingConfig(
                    include_thoughts=True, thinking_budget=0
                ),
                temperature=TEMPERATURE,
            ),
        )
        return response.text
    elif model_name != "LiquidAI/LFM2-700M":
        messages = [{"role": "user", "content": prompt}]
        if "gpt-oss" in model_name:
            messages.insert(0, {"role": "system", "content": "Reasoning: low"})

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
        )

        return completion.choices[0].message.content
