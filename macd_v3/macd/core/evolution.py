
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from .utils import deep_merge


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class MutationType(Enum):
    """Mutation types for genetic algorithm."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


@dataclass
class EvolutionConfig:
    """Configuration for evolution algorithm."""
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    mutation_type: MutationType = MutationType.GAUSSIAN
    adaptive_mutation: bool = True
    diversity_threshold: float = 0.1
    max_generations: int = 100
    convergence_threshold: float = 1e-6
    convergence_window: int = 10


GENE_SPACE = {
    "generation": {
        "temperature": (0.0, 2.0),
        "top_p": (0.1, 1.0),
        "max_new_tokens": (16, 512),
        "repetition_penalty": (1.0, 1.5),
    }
}

PROMPT_STYLES = ["cot", "concise", "bullet", "json", "detailed", "structured"]

# Legacy gene space kept for backward compatibility but marked as deprecated
# These parameters are not currently used by the models
DEPRECATED_GENE_SPACE = {
    "retrieval": {
        "k": (1, 20),
        "chunk_size": (100, 2000),
        "similarity_threshold": (0.0, 1.0),
    },
    "training": {
        "lr": (1e-6, 1e-3),
        "rank": (4, 64),
        "steps": (50, 2000),
        "batch_size": (1, 16),
    }
}


def random_strategy(seed: Optional[int] = None, include_deprecated: bool = False) -> Dict[str, Any]:
    """
    Generate a random strategy within the gene space.

    Args:
        seed: Random seed for reproducibility
        include_deprecated: If True, include deprecated retrieval/training parameters
                          (for backward compatibility only)

    Returns:
        Dictionary containing strategy parameters
    """
    rng = random.Random(seed)
    strategy = {
        "generation": {
            "temperature": round(rng.uniform(*GENE_SPACE["generation"]["temperature"]), 3),
            "top_p": round(rng.uniform(*GENE_SPACE["generation"]["top_p"]), 3),
            "max_new_tokens": rng.randint(*GENE_SPACE["generation"]["max_new_tokens"]),
            "repetition_penalty": round(rng.uniform(*GENE_SPACE["generation"]["repetition_penalty"]), 3),
        },
        "prompt_style": rng.choice(PROMPT_STYLES),
    }

    # Add deprecated parameters only if explicitly requested
    if include_deprecated:
        strategy["retrieval"] = {
            "k": rng.randint(*DEPRECATED_GENE_SPACE["retrieval"]["k"]),
            "chunk_size": rng.randint(*DEPRECATED_GENE_SPACE["retrieval"]["chunk_size"]),
            "similarity_threshold": round(rng.uniform(*DEPRECATED_GENE_SPACE["retrieval"]["similarity_threshold"]), 3),
        }
        strategy["training"] = {
            "lr": round(rng.uniform(*DEPRECATED_GENE_SPACE["training"]["lr"]), 8),
            "rank": rng.randint(*DEPRECATED_GENE_SPACE["training"]["rank"]),
            "steps": rng.randint(*DEPRECATED_GENE_SPACE["training"]["steps"]),
            "batch_size": rng.randint(*DEPRECATED_GENE_SPACE["training"]["batch_size"]),
        }

    return strategy


def tournament_selection(population: List[Tuple[float, Dict[str, Any]]], 
                        tournament_size: int, 
                        rng: random.Random) -> Dict[str, Any]:
    """Select individual using tournament selection."""
    tournament = rng.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x[0])[1]


def roulette_selection(population: List[Tuple[float, Dict[str, Any]]], 
                      rng: random.Random) -> Dict[str, Any]:
    """Select individual using roulette wheel selection."""
    # Normalize fitness scores to positive values
    min_fitness = min(score for score, _ in population)
    adjusted_scores = [score - min_fitness + 1e-6 for score, _ in population]
    total_fitness = sum(adjusted_scores)
    
    if total_fitness == 0:
        return rng.choice(population)[1]
    
    # Roulette wheel selection
    r = rng.uniform(0, total_fitness)
    cumulative = 0
    for score, individual in zip(adjusted_scores, population):
        cumulative += score
        if cumulative >= r:
            return individual[1]
    
    return population[-1][1]


def rank_selection(population: List[Tuple[float, Dict[str, Any]]], 
                  rng: random.Random) -> Dict[str, Any]:
    """Select individual using rank-based selection."""
    # Sort by fitness (descending)
    sorted_pop = sorted(population, key=lambda x: x[0], reverse=True)
    n = len(sorted_pop)
    
    # Linear ranking: rank 1 gets weight n, rank n gets weight 1
    weights = [n - i for i in range(n)]
    total_weight = sum(weights)
    
    r = rng.uniform(0, total_weight)
    cumulative = 0
    for i, (score, individual) in enumerate(sorted_pop):
        cumulative += weights[i]
        if cumulative >= r:
            return individual[1]
    
    return sorted_pop[-1][1]


def gaussian_mutation(strategy: Dict[str, Any],
                     strength: float = 0.1,
                     rng: random.Random = None) -> Dict[str, Any]:
    """
    Apply Gaussian mutation to strategy.

    Only mutates generation parameters and prompt_style since other parameters
    are not currently used by models.
    """
    if rng is None:
        rng = random.Random()

    child = deep_merge({}, strategy)

    # Ensure generation section exists
    if "generation" not in child:
        child["generation"] = {}

    # Mutate generation parameters
    for param, (min_val, max_val) in GENE_SPACE["generation"].items():
        if param in child["generation"]:
            current = child["generation"][param]
            if isinstance(current, (int, float)):
                noise = rng.gauss(0, strength * (max_val - min_val))
                new_val = current + noise
                if isinstance(current, int):
                    new_val = int(round(new_val))
                child["generation"][param] = max(min_val, min(max_val, new_val))

    # Mutate deprecated parameters only if they exist (backward compatibility)
    for section in ["retrieval", "training"]:
        if section in child and section in DEPRECATED_GENE_SPACE:
            for param, (min_val, max_val) in DEPRECATED_GENE_SPACE[section].items():
                if param in child[section]:
                    current = child[section][param]
                    if isinstance(current, (int, float)):
                        noise = rng.gauss(0, strength * (max_val - min_val))
                        new_val = current + noise
                        if isinstance(current, int):
                            new_val = int(round(new_val))
                        child[section][param] = max(min_val, min(max_val, new_val))

    # Mutate prompt style
    if rng.random() < 0.1:  # 10% chance to change prompt style
        child["prompt_style"] = rng.choice(PROMPT_STYLES)

    return child


def uniform_mutation(strategy: Dict[str, Any], 
                    strength: float = 0.1, 
                    rng: random.Random = None) -> Dict[str, Any]:
    """Apply uniform mutation to strategy."""
    if rng is None:
        rng = random.Random()
    
    child = deep_merge({}, strategy)
    
    # Mutate each parameter with uniform noise
    for section, params in [("generation", GENE_SPACE["generation"]), 
                           ("retrieval", GENE_SPACE["retrieval"]), 
                           ("training", GENE_SPACE["training"])]:
        for param, (min_val, max_val) in params.items():
            if param in child[section]:
                current = child[section][param]
                if isinstance(current, (int, float)):
                    noise = rng.uniform(-strength, strength) * (max_val - min_val)
                    new_val = current + noise
                    if isinstance(current, int):
                        new_val = int(round(new_val))
                    child[section][param] = max(min_val, min(max_val, new_val))
    
    # Mutate prompt style
    if rng.random() < 0.1:
        child["prompt_style"] = rng.choice(PROMPT_STYLES)
    
    return child


def adaptive_mutation(strategy: Dict[str, Any], 
                     generation: int, 
                     max_generations: int,
                     diversity: float,
                     rng: random.Random = None) -> Dict[str, Any]:
    """Apply adaptive mutation based on generation and diversity."""
    if rng is None:
        rng = random.Random()
    
    # Adaptive mutation rate based on generation and diversity
    base_rate = 0.1
    generation_factor = 1.0 - (generation / max_generations)  # Decrease over time
    diversity_factor = 1.0 - diversity  # Increase when diversity is low
    
    adaptive_strength = base_rate * generation_factor * diversity_factor
    adaptive_strength = max(0.01, min(0.5, adaptive_strength))  # Clamp to reasonable range
    
    return gaussian_mutation(strategy, adaptive_strength, rng)


def uniform_crossover(parent1: Dict[str, Any], 
                     parent2: Dict[str, Any], 
                     rng: random.Random = None) -> Dict[str, Any]:
    """Perform uniform crossover between two strategies."""
    if rng is None:
        rng = random.Random()
    
    child = {}
    
    # Crossover each section
    for section in ["generation", "retrieval", "training"]:
        child[section] = {}
        for key in set(parent1.get(section, {}).keys()) | set(parent2.get(section, {}).keys()):
            if rng.random() < 0.5:
                child[section][key] = parent1.get(section, {}).get(key)
            else:
                child[section][key] = parent2.get(section, {}).get(key)
    
    # Crossover prompt style
    child["prompt_style"] = rng.choice([parent1.get("prompt_style", "concise"), 
                                      parent2.get("prompt_style", "concise")])
    
    return child


def arithmetic_crossover(parent1: Dict[str, Any], 
                        parent2: Dict[str, Any], 
                        alpha: float = 0.5,
                        rng: random.Random = None) -> Dict[str, Any]:
    """Perform arithmetic crossover for numerical parameters."""
    if rng is None:
        rng = random.Random()
    
    child = deep_merge({}, parent1)
    
    # Arithmetic crossover for numerical parameters
    for section in ["generation", "retrieval", "training"]:
        if section in parent1 and section in parent2:
            for key in parent1[section]:
                if key in parent2[section]:
                    val1 = parent1[section][key]
                    val2 = parent2[section][key]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        new_val = alpha * val1 + (1 - alpha) * val2
                        if isinstance(val1, int):
                            new_val = int(round(new_val))
                        child[section][key] = new_val
    
    return child


def calculate_diversity(population: List[Dict[str, Any]]) -> float:
    """Calculate population diversity based on parameter variance."""
    if len(population) < 2:
        return 0.0
    
    # Extract numerical parameters
    params = []
    for individual in population:
        individual_params = []
        for section in ["generation", "retrieval", "training"]:
            if section in individual:
                for key, value in individual[section].items():
                    if isinstance(value, (int, float)):
                        individual_params.append(value)
        params.append(individual_params)
    
    if not params or not params[0]:
        return 0.0
    
    # Calculate average variance across parameters
    param_array = np.array(params)
    variances = np.var(param_array, axis=0)
    avg_variance = np.mean(variances)
    
    # Normalize to [0, 1] range
    return min(1.0, avg_variance)


class EvolutionEngine:
    """Advanced evolution engine with multiple algorithms and operators."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.rng = random.Random()
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
    
    def evolve(self, 
               population: List[Tuple[float, Dict[str, Any]]], 
               fitness_function: callable) -> Dict[str, Any]:
        """
        Evolve population for one generation.
        
        Args:
            population: List of (fitness, strategy) tuples
            fitness_function: Function to evaluate strategy fitness
            
        Returns:
            Dictionary with evolution results
        """
        if len(population) < 2:
            raise ValueError("Population must have at least 2 individuals")
        
        # Sort population by fitness (descending)
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Track best fitness
        best_fitness = population[0][0]
        self.best_fitness_history.append(best_fitness)
        
        # Calculate diversity
        strategies = [strategy for _, strategy in population]
        diversity = calculate_diversity(strategies)
        self.diversity_history.append(diversity)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = min(self.config.elite_size, len(population))
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent(population)
            parent2 = self._select_parent(population)
            
            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                if self.rng.random() < 0.5:
                    child = uniform_crossover(parent1, parent2, self.rng)
                else:
                    child = arithmetic_crossover(parent1, parent2, self.rng.uniform(0.3, 0.7), self.rng)
            else:
                child = deep_merge({}, parent1)
            
            # Mutation
            if self.rng.random() < self.config.mutation_rate:
                if self.config.mutation_type == MutationType.GAUSSIAN:
                    child = gaussian_mutation(child, 0.1, self.rng)
                elif self.config.mutation_type == MutationType.UNIFORM:
                    child = uniform_mutation(child, 0.1, self.rng)
                elif self.config.mutation_type == MutationType.ADAPTIVE:
                    child = adaptive_mutation(child, self.generation, self.config.max_generations, 
                                            diversity, self.rng)
            
            # Evaluate child
            child_fitness = fitness_function(child)
            new_population.append((child_fitness, child))
        
        self.generation += 1
        
        return {
            "population": new_population,
            "best_fitness": best_fitness,
            "diversity": diversity,
            "generation": self.generation,
            "converged": self._check_convergence()
        }
    
    def _select_parent(self, population: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, Any]:
        """Select parent using configured selection method."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return tournament_selection(population, self.config.tournament_size, self.rng)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return roulette_selection(population, self.rng)
        elif self.config.selection_method == SelectionMethod.RANK:
            return rank_selection(population, self.rng)
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.best_fitness_history) < self.config.convergence_window:
            return False
        
        recent_fitness = self.best_fitness_history[-self.config.convergence_window:]
        fitness_std = np.std(recent_fitness)
        
        return fitness_std < self.config.convergence_threshold


# Backward compatibility functions
def mutate(strategy: Dict[str, Any], strength: float = 0.2) -> Dict[str, Any]:
    """Backward compatibility function for mutation."""
    return gaussian_mutation(strategy, strength)


def crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility function for crossover."""
    return uniform_crossover(a, b)
