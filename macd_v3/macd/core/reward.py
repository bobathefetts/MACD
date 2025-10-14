
from typing import Dict, Any
# Placeholder for a learned reward model.
# In a real system you'd encode (prompt, prediction, reference) and score them.
def simple_reward(prompt: str, prediction: str, reference: str) -> float:
    # Reward is higher when prediction token overlap is higher with reference
    from .evaluator import f1_unigram
    return f1_unigram(prediction, reference)
