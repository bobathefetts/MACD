"""Tests for evolution algorithms."""

import pytest
import numpy as np
from typing import Dict, Any

from macd.core.evolution import (
    random_strategy, 
    gaussian_mutation, 
    uniform_mutation,
    adaptive_mutation,
    uniform_crossover,
    arithmetic_crossover,
    tournament_selection,
    roulette_selection,
    rank_selection,
    calculate_diversity,
    EvolutionEngine,
    EvolutionConfig,
    SelectionMethod,
    MutationType,
    GENE_SPACE,
    PROMPT_STYLES
)


class TestStrategyGeneration:
    """Test strategy generation functions."""
    
    def test_random_strategy(self):
        """Test random strategy generation."""
        strategy = random_strategy(seed=42)
        
        # Check structure
        assert "generation" in strategy
        assert "retrieval" in strategy
        assert "training" in strategy
        assert "prompt_style" in strategy
        
        # Check generation parameters
        gen_params = strategy["generation"]
        assert "temperature" in gen_params
        assert "top_p" in gen_params
        assert "max_new_tokens" in gen_params
        assert "repetition_penalty" in gen_params
        
        # Check parameter bounds
        assert 0.0 <= gen_params["temperature"] <= 2.0
        assert 0.1 <= gen_params["top_p"] <= 1.0
        assert 16 <= gen_params["max_new_tokens"] <= 512
        assert 1.0 <= gen_params["repetition_penalty"] <= 1.5
        
        # Check retrieval parameters
        ret_params = strategy["retrieval"]
        assert "k" in ret_params
        assert "chunk_size" in ret_params
        assert "similarity_threshold" in ret_params
        
        assert 1 <= ret_params["k"] <= 20
        assert 100 <= ret_params["chunk_size"] <= 2000
        assert 0.0 <= ret_params["similarity_threshold"] <= 1.0
        
        # Check training parameters
        train_params = strategy["training"]
        assert "lr" in train_params
        assert "rank" in train_params
        assert "steps" in train_params
        assert "batch_size" in train_params
        
        assert 1e-6 <= train_params["lr"] <= 1e-3
        assert 4 <= train_params["rank"] <= 64
        assert 50 <= train_params["steps"] <= 2000
        assert 1 <= train_params["batch_size"] <= 16
        
        # Check prompt style
        assert strategy["prompt_style"] in PROMPT_STYLES
    
    def test_random_strategy_deterministic(self):
        """Test that random strategy is deterministic with same seed."""
        strategy1 = random_strategy(seed=42)
        strategy2 = random_strategy(seed=42)
        
        assert strategy1 == strategy2
    
    def test_random_strategy_different_seeds(self):
        """Test that different seeds produce different strategies."""
        strategy1 = random_strategy(seed=42)
        strategy2 = random_strategy(seed=123)
        
        assert strategy1 != strategy2


class TestMutation:
    """Test mutation functions."""
    
    def test_gaussian_mutation(self):
        """Test Gaussian mutation."""
        strategy = random_strategy(seed=42)
        mutated = gaussian_mutation(strategy, strength=0.1, seed=42)
        
        # Check that structure is preserved
        assert set(strategy.keys()) == set(mutated.keys())
        for section in ["generation", "retrieval", "training"]:
            assert set(strategy[section].keys()) == set(mutated[section].keys())
        
        # Check that values are within bounds
        for section, params in GENE_SPACE.items():
            for param, (min_val, max_val) in params.items():
                if param in mutated[section]:
                    assert min_val <= mutated[section][param] <= max_val
    
    def test_uniform_mutation(self):
        """Test uniform mutation."""
        strategy = random_strategy(seed=42)
        mutated = uniform_mutation(strategy, strength=0.1, seed=42)
        
        # Check that structure is preserved
        assert set(strategy.keys()) == set(mutated.keys())
        
        # Check that values are within bounds
        for section, params in GENE_SPACE.items():
            for param, (min_val, max_val) in params.items():
                if param in mutated[section]:
                    assert min_val <= mutated[section][param] <= max_val
    
    def test_adaptive_mutation(self):
        """Test adaptive mutation."""
        strategy = random_strategy(seed=42)
        mutated = adaptive_mutation(
            strategy, 
            generation=10, 
            max_generations=100, 
            diversity=0.5, 
            seed=42
        )
        
        # Check that structure is preserved
        assert set(strategy.keys()) == set(mutated.keys())
        
        # Check that values are within bounds
        for section, params in GENE_SPACE.items():
            for param, (min_val, max_val) in params.items():
                if param in mutated[section]:
                    assert min_val <= mutated[section][param] <= max_val
    
    def test_mutation_preserves_types(self):
        """Test that mutation preserves data types."""
        strategy = random_strategy(seed=42)
        mutated = gaussian_mutation(strategy, strength=0.1)
        
        # Check integer parameters remain integers
        assert isinstance(mutated["generation"]["max_new_tokens"], int)
        assert isinstance(mutated["retrieval"]["k"], int)
        assert isinstance(mutated["training"]["rank"], int)
        assert isinstance(mutated["training"]["steps"], int)
        assert isinstance(mutated["training"]["batch_size"], int)
        
        # Check float parameters remain floats
        assert isinstance(mutated["generation"]["temperature"], float)
        assert isinstance(mutated["generation"]["top_p"], float)
        assert isinstance(mutated["retrieval"]["similarity_threshold"], float)
        assert isinstance(mutated["training"]["lr"], float)


class TestCrossover:
    """Test crossover functions."""
    
    def test_uniform_crossover(self):
        """Test uniform crossover."""
        parent1 = random_strategy(seed=42)
        parent2 = random_strategy(seed=123)
        
        child = uniform_crossover(parent1, parent2, seed=42)
        
        # Check that structure is preserved
        assert set(parent1.keys()) == set(child.keys())
        for section in ["generation", "retrieval", "training"]:
            assert set(parent1[section].keys()) == set(child[section].keys())
        
        # Check that child has values from both parents
        # (This is probabilistic, so we just check structure)
        assert "prompt_style" in child
    
    def test_arithmetic_crossover(self):
        """Test arithmetic crossover."""
        parent1 = random_strategy(seed=42)
        parent2 = random_strategy(seed=123)
        
        child = arithmetic_crossover(parent1, parent2, alpha=0.5, seed=42)
        
        # Check that structure is preserved
        assert set(parent1.keys()) == set(child.keys())
        
        # Check that numerical values are interpolated
        for section in ["generation", "retrieval", "training"]:
            for param in parent1[section]:
                if param in parent2[section]:
                    val1 = parent1[section][param]
                    val2 = parent2[section][param]
                    child_val = child[section][param]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Child value should be between parent values (approximately)
                        min_val = min(val1, val2)
                        max_val = max(val1, val2)
                        assert min_val <= child_val <= max_val


class TestSelection:
    """Test selection functions."""
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = [
            (0.9, {"fitness": 0.9}),
            (0.7, {"fitness": 0.7}),
            (0.5, {"fitness": 0.5}),
            (0.3, {"fitness": 0.3}),
            (0.1, {"fitness": 0.1})
        ]
        
        import random
        rng = random.Random(42)
        
        # Run multiple times to test selection
        selected = tournament_selection(population, tournament_size=3, rng=rng)
        assert selected in [ind[1] for ind in population]
    
    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        population = [
            (0.9, {"fitness": 0.9}),
            (0.7, {"fitness": 0.7}),
            (0.5, {"fitness": 0.5}),
            (0.3, {"fitness": 0.3}),
            (0.1, {"fitness": 0.1})
        ]
        
        import random
        rng = random.Random(42)
        
        selected = roulette_selection(population, rng)
        assert selected in [ind[1] for ind in population]
    
    def test_rank_selection(self):
        """Test rank-based selection."""
        population = [
            (0.9, {"fitness": 0.9}),
            (0.7, {"fitness": 0.7}),
            (0.5, {"fitness": 0.5}),
            (0.3, {"fitness": 0.3}),
            (0.1, {"fitness": 0.1})
        ]
        
        import random
        rng = random.Random(42)
        
        selected = rank_selection(population, rng)
        assert selected in [ind[1] for ind in population]


class TestDiversity:
    """Test diversity calculation."""
    
    def test_calculate_diversity(self):
        """Test diversity calculation."""
        # Identical strategies should have zero diversity
        strategy = random_strategy(seed=42)
        identical_population = [strategy, strategy, strategy]
        diversity = calculate_diversity(identical_population)
        assert diversity == 0.0
        
        # Different strategies should have positive diversity
        different_population = [
            random_strategy(seed=42),
            random_strategy(seed=123),
            random_strategy(seed=456)
        ]
        diversity = calculate_diversity(different_population)
        assert diversity > 0.0
        assert diversity <= 1.0
    
    def test_calculate_diversity_single_individual(self):
        """Test diversity calculation with single individual."""
        population = [random_strategy(seed=42)]
        diversity = calculate_diversity(population)
        assert diversity == 0.0
    
    def test_calculate_diversity_empty_population(self):
        """Test diversity calculation with empty population."""
        population = []
        diversity = calculate_diversity(population)
        assert diversity == 0.0


class TestEvolutionEngine:
    """Test EvolutionEngine class."""
    
    def test_evolution_engine_initialization(self):
        """Test evolution engine initialization."""
        config = EvolutionConfig(
            population_size=20,
            elite_size=3,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        engine = EvolutionEngine(config)
        assert engine.config.population_size == 20
        assert engine.config.elite_size == 3
        assert engine.generation == 0
        assert len(engine.best_fitness_history) == 0
    
    def test_evolution_engine_evolve(self):
        """Test evolution engine evolve method."""
        config = EvolutionConfig(
            population_size=10,
            elite_size=2,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        engine = EvolutionEngine(config)
        
        # Create initial population
        population = [
            (0.8, random_strategy(seed=42)),
            (0.7, random_strategy(seed=123)),
            (0.6, random_strategy(seed=456)),
            (0.5, random_strategy(seed=789)),
            (0.4, random_strategy(seed=101))
        ]
        
        # Mock fitness function
        def fitness_function(strategy):
            return np.random.random()
        
        # Run evolution
        result = engine.evolve(population, fitness_function)
        
        # Check result structure
        assert "population" in result
        assert "best_fitness" in result
        assert "diversity" in result
        assert "generation" in result
        assert "converged" in result
        
        # Check population size
        assert len(result["population"]) == config.population_size
        
        # Check generation increment
        assert engine.generation == 1
        
        # Check fitness history
        assert len(engine.best_fitness_history) == 1
    
    def test_evolution_engine_convergence(self):
        """Test evolution engine convergence detection."""
        config = EvolutionConfig(
            population_size=10,
            elite_size=2,
            convergence_threshold=0.01,
            convergence_window=3
        )
        
        engine = EvolutionEngine(config)
        
        # Simulate convergence
        engine.best_fitness_history = [0.8, 0.81, 0.82, 0.83, 0.84]
        
        # Should not converge yet (not enough generations)
        assert not engine._check_convergence()
        
        # Add more generations with small improvements
        engine.best_fitness_history.extend([0.85, 0.86, 0.87, 0.88, 0.89, 0.90])
        
        # Should converge now
        assert engine._check_convergence()
    
    def test_evolution_engine_insufficient_population(self):
        """Test evolution engine with insufficient population."""
        config = EvolutionConfig(population_size=10, elite_size=2)
        engine = EvolutionEngine(config)
        
        # Population with only one individual
        population = [(0.8, random_strategy(seed=42))]
        
        def fitness_function(strategy):
            return 0.5
        
        with pytest.raises(ValueError, match="Population must have at least 2 individuals"):
            engine.evolve(population, fitness_function)
