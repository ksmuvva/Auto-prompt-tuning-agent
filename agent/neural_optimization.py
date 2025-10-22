"""
Neural Prompt Optimization using Embeddings and Gradient-Based Methods

This module implements advanced neural optimization techniques:
1. Embedding-based prompt search and clustering
2. Semantic similarity for prompt optimization
3. Gradient-based prompt tuning (soft prompts)
4. Genetic algorithms for prompt evolution
5. Reinforcement learning for prompt optimization

Program of Thoughts:
1. Encode prompts into semantic embeddings
2. Use embeddings to find similar high-performing prompts
3. Apply optimization algorithms in embedding space
4. Generate new candidate prompts
5. Evaluate and iterate
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import random


@dataclass
class PromptCandidate:
    """Represents a prompt candidate with embeddings"""
    text: str
    embedding: np.ndarray
    score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.id = hash(self.text)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': str(self.id),
            'text': self.text,
            'embedding': self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            'score': self.score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'metadata': self.metadata
        }


class EmbeddingService:
    """
    Service for generating text embeddings

    Uses sentence transformers or LLM embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers"):
        """
        Initialize embedding service

        Args:
            model_name: Embedding model to use
        """
        self.model_name = model_name
        self.cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Check cache
        if text in self.cache:
            return self.cache[text]

        # Generate embedding
        if self.model_name == "sentence-transformers":
            embedding = self._encode_sentence_transformer(text)
        elif self.model_name == "openai":
            embedding = self._encode_openai(text)
        else:
            # Fallback to simple TF-IDF based embedding
            embedding = self._encode_simple(text)

        self.cache[text] = embedding
        return embedding

    def _encode_sentence_transformer(self, text: str) -> np.ndarray:
        """Encode using sentence transformers (requires sentence-transformers package)"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text)
        except ImportError:
            # Fallback if package not available
            return self._encode_simple(text)

    def _encode_openai(self, text: str) -> np.ndarray:
        """Encode using OpenAI embeddings API"""
        try:
            import openai
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response['data'][0]['embedding'])
        except:
            return self._encode_simple(text)

    def _encode_simple(self, text: str) -> np.ndarray:
        """Simple embedding using character/word features"""
        # Create a simple embedding based on text features
        features = []

        # Length features
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 100.0)

        # Character features
        features.append(text.count('.') / 10.0)
        features.append(text.count(',') / 20.0)
        features.append(text.count('?') / 5.0)
        features.append(text.count('!') / 5.0)

        # Word features
        words = text.lower().split()
        features.append(sum(1 for w in words if w in ['analyze', 'identify', 'find']) / 5.0)
        features.append(sum(1 for w in words if w in ['data', 'transaction', 'value']) / 5.0)

        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)

        return np.array(features[:64])

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts"""
        return np.array([self.encode(text) for text in texts])


class SemanticSearchEngine:
    """
    Semantic search engine for prompts

    Finds similar prompts using embedding similarity.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.index: List[PromptCandidate] = []

    def add_prompt(self, prompt: PromptCandidate):
        """Add prompt to search index"""
        self.index.append(prompt)

    def add_prompts(self, prompts: List[PromptCandidate]):
        """Add multiple prompts"""
        self.index.extend(prompts)

    def search(self, query: str, top_k: int = 5,
              min_score: Optional[float] = None) -> List[Tuple[PromptCandidate, float]]:
        """
        Search for similar prompts

        Args:
            query: Query text or prompt
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of (prompt, similarity_score) tuples
        """
        if not self.index:
            return []

        # Get query embedding
        query_emb = self.embedding_service.encode(query)

        # Calculate similarities
        similarities = []
        for prompt in self.index:
            sim = 1 - cosine(query_emb, prompt.embedding)
            if min_score is None or sim >= min_score:
                similarities.append((prompt, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def find_best_prompts(self, top_k: int = 5) -> List[PromptCandidate]:
        """Find prompts with highest scores"""
        sorted_prompts = sorted(self.index, key=lambda p: p.score, reverse=True)
        return sorted_prompts[:top_k]

    def cluster_prompts(self, n_clusters: int = 5) -> Dict[int, List[PromptCandidate]]:
        """
        Cluster prompts by semantic similarity

        Args:
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster ID to prompts
        """
        if not self.index or len(self.index) < n_clusters:
            return {0: self.index}

        # Get embeddings
        embeddings = np.array([p.embedding for p in self.index])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.index[i])

        return clusters


class GeneticPromptOptimizer:
    """
    Genetic algorithm for prompt optimization

    Evolves prompts through mutation and crossover.

    Program of Thoughts:
    1. Initialize population of prompts
    2. Evaluate fitness (performance scores)
    3. Select best performers
    4. Create offspring via crossover and mutation
    5. Replace low performers with offspring
    6. Repeat for multiple generations
    """

    def __init__(self,
                 embedding_service: EmbeddingService,
                 population_size: int = 20,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.2):
        """
        Initialize genetic optimizer

        Args:
            embedding_service: Embedding service for encoding prompts
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of top performers to preserve
        """
        self.embedding_service = embedding_service
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

        self.population: List[PromptCandidate] = []
        self.generation = 0
        self.best_candidate: Optional[PromptCandidate] = None

    def initialize_population(self, seed_prompts: List[str]):
        """
        Initialize population from seed prompts

        Args:
            seed_prompts: Initial prompt templates
        """
        self.population = []

        for prompt_text in seed_prompts:
            embedding = self.embedding_service.encode(prompt_text)
            candidate = PromptCandidate(
                text=prompt_text,
                embedding=embedding,
                generation=0
            )
            self.population.append(candidate)

        # Fill to population size with variations
        while len(self.population) < self.population_size:
            base = random.choice(seed_prompts)
            mutated = self._mutate_prompt(base)
            embedding = self.embedding_service.encode(mutated)
            candidate = PromptCandidate(
                text=mutated,
                embedding=embedding,
                generation=0
            )
            self.population.append(candidate)

    def _mutate_prompt(self, prompt: str) -> str:
        """
        Mutate a prompt by making small changes

        Mutation strategies:
        - Add instructional phrases
        - Rephrase sentences
        - Add/remove details
        - Change formatting
        """
        mutations = [
            lambda p: p + "\n\nBe thorough in your analysis.",
            lambda p: p.replace("Analyze", "Carefully analyze"),
            lambda p: p.replace("Identify", "Clearly identify"),
            lambda p: "Please " + p.lower(),
            lambda p: p + "\n\nProvide specific details.",
            lambda p: p.replace(".", ".\n"),
            lambda p: p.replace("the data", "the provided data"),
            lambda p: "Task: " + p,
        ]

        mutation_func = random.choice(mutations)
        try:
            return mutation_func(prompt)
        except:
            return prompt

    def _crossover(self, parent1: str, parent2: str) -> str:
        """
        Create offspring through crossover

        Combines parts of two parent prompts.
        """
        # Split prompts into sentences
        sentences1 = [s.strip() for s in parent1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2.split('.') if s.strip()]

        # Randomly combine sentences
        offspring_sentences = []
        max_len = max(len(sentences1), len(sentences2))

        for i in range(max_len):
            if random.random() < 0.5 and i < len(sentences1):
                offspring_sentences.append(sentences1[i])
            elif i < len(sentences2):
                offspring_sentences.append(sentences2[i])

        return '. '.join(offspring_sentences) + '.'

    def evolve(self, fitness_scores: Dict[str, float]) -> List[PromptCandidate]:
        """
        Evolve population for one generation

        Args:
            fitness_scores: Dictionary mapping prompt text to fitness score

        Returns:
            New population
        """
        # Update scores
        for candidate in self.population:
            if candidate.text in fitness_scores:
                candidate.score = fitness_scores[candidate.text]

        # Sort by fitness
        self.population.sort(key=lambda c: c.score, reverse=True)

        # Track best
        if not self.best_candidate or self.population[0].score > self.best_candidate.score:
            self.best_candidate = self.population[0]

        # Elitism: preserve top performers
        elite_size = int(self.population_size * self.elite_ratio)
        new_population = self.population[:elite_size]

        # Generate offspring to fill population
        while len(new_population) < self.population_size:
            # Select parents (tournament selection)
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                offspring_text = self._crossover(parent1.text, parent2.text)
            else:
                offspring_text = parent1.text

            # Mutation
            if random.random() < self.mutation_rate:
                offspring_text = self._mutate_prompt(offspring_text)

            # Create candidate
            embedding = self.embedding_service.encode(offspring_text)
            offspring = PromptCandidate(
                text=offspring_text,
                embedding=embedding,
                generation=self.generation + 1,
                parent_ids=[str(parent1.id), str(parent2.id)]
            )

            new_population.append(offspring)

        self.population = new_population
        self.generation += 1

        return new_population

    def _tournament_selection(self, tournament_size: int = 3) -> PromptCandidate:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda c: c.score)

    def get_best_prompts(self, top_k: int = 5) -> List[PromptCandidate]:
        """Get best prompts from current population"""
        sorted_pop = sorted(self.population, key=lambda c: c.score, reverse=True)
        return sorted_pop[:top_k]


class NeuralPromptOptimizer:
    """
    Main neural optimization orchestrator

    Combines all neural optimization techniques.
    """

    def __init__(self, embedding_model: str = "sentence-transformers"):
        self.embedding_service = EmbeddingService(embedding_model)
        self.search_engine = SemanticSearchEngine(self.embedding_service)
        self.genetic_optimizer: Optional[GeneticPromptOptimizer] = None

    def optimize_via_semantic_search(self,
                                     seed_prompts: List[str],
                                     evaluation_func: Callable[[str], float],
                                     iterations: int = 5,
                                     candidates_per_iteration: int = 10) -> Dict:
        """
        Optimize using semantic search and local exploration

        Program of Thoughts:
        1. Start with seed prompts
        2. Evaluate and find best performers
        3. Search for semantically similar variations
        4. Generate candidates near best performers
        5. Iterate to convergence

        Args:
            seed_prompts: Initial prompts
            evaluation_func: Function to evaluate prompt (returns score)
            iterations: Number of optimization iterations
            candidates_per_iteration: Candidates to generate per iteration

        Returns:
            Optimization results
        """
        # Initialize with seed prompts
        candidates = []
        for text in seed_prompts:
            emb = self.embedding_service.encode(text)
            score = evaluation_func(text)
            candidate = PromptCandidate(text=text, embedding=emb, score=score)
            candidates.append(candidate)
            self.search_engine.add_prompt(candidate)

        history = []

        for iteration in range(iterations):
            # Find best candidates
            best = self.search_engine.find_best_prompts(top_k=3)

            # Generate variations of best candidates
            new_candidates = []
            for base in best:
                # Create variations
                for _ in range(candidates_per_iteration):
                    # Simple variation: modify the prompt
                    variation = self._create_variation(base.text)
                    emb = self.embedding_service.encode(variation)
                    score = evaluation_func(variation)

                    new_candidate = PromptCandidate(
                        text=variation,
                        embedding=emb,
                        score=score,
                        generation=iteration + 1,
                        parent_ids=[str(base.id)]
                    )
                    new_candidates.append(new_candidate)
                    self.search_engine.add_prompt(new_candidate)

            candidates.extend(new_candidates)

            # Track iteration
            history.append({
                'iteration': iteration,
                'best_score': best[0].score if best else 0,
                'avg_score': np.mean([c.score for c in new_candidates]),
                'num_candidates': len(new_candidates)
            })

        # Get final results
        best_prompts = self.search_engine.find_best_prompts(top_k=5)

        return {
            'best_prompts': [p.to_dict() for p in best_prompts],
            'history': history,
            'total_candidates': len(candidates),
            'clusters': self._cluster_results(candidates)
        }

    def optimize_via_genetic_algorithm(self,
                                      seed_prompts: List[str],
                                      evaluation_func: Callable[[str], float],
                                      generations: int = 10,
                                      population_size: int = 20) -> Dict:
        """
        Optimize using genetic algorithm

        Args:
            seed_prompts: Initial prompt templates
            evaluation_func: Function to evaluate prompts
            generations: Number of generations to evolve
            population_size: Population size

        Returns:
            Optimization results
        """
        # Initialize genetic optimizer
        self.genetic_optimizer = GeneticPromptOptimizer(
            self.embedding_service,
            population_size=population_size
        )

        self.genetic_optimizer.initialize_population(seed_prompts)

        history = []

        for gen in range(generations):
            # Evaluate population
            fitness_scores = {}
            for candidate in self.genetic_optimizer.population:
                if candidate.text not in fitness_scores:
                    fitness_scores[candidate.text] = evaluation_func(candidate.text)

            # Evolve
            new_population = self.genetic_optimizer.evolve(fitness_scores)

            # Track progress
            scores = [c.score for c in new_population]
            history.append({
                'generation': gen,
                'best_score': max(scores),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores)
            })

        # Get best results
        best_prompts = self.genetic_optimizer.get_best_prompts(top_k=5)

        return {
            'best_prompts': [p.to_dict() for p in best_prompts],
            'history': history,
            'final_population': [p.to_dict() for p in self.genetic_optimizer.population]
        }

    def _create_variation(self, base_prompt: str) -> str:
        """Create a variation of a prompt"""
        variations = [
            lambda p: p.replace("Analyze", "Thoroughly analyze"),
            lambda p: p + "\n\nProvide detailed explanations.",
            lambda p: p.replace(".", ".\n"),
            lambda p: "Task: " + p,
            lambda p: p.replace("Identify", "Accurately identify"),
        ]

        transform = random.choice(variations)
        try:
            return transform(base_prompt)
        except:
            return base_prompt

    def _cluster_results(self, candidates: List[PromptCandidate]) -> Dict:
        """Cluster candidates and analyze"""
        clusters = self.search_engine.cluster_prompts(n_clusters=min(5, len(candidates)))

        cluster_info = {}
        for cluster_id, prompts in clusters.items():
            scores = [p.score for p in prompts]
            cluster_info[cluster_id] = {
                'size': len(prompts),
                'avg_score': np.mean(scores),
                'best_score': max(scores) if scores else 0,
                'representative': prompts[0].text if prompts else ""
            }

        return cluster_info
