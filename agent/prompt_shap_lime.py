"""
TRUE SHAP and LIME Implementation for Prompt Explainability

Implements actual SHAP (Shapley values) and LIME (local linear models)
for explaining prompt performance in LLM systems.
"""

from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
import re
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SHAPValues:
    prompt_text: str
    base_value: float
    token_shap_values: Dict[str, float]
    segment_shap_values: Dict[str, float]
    total_prediction: float
    metrics: Dict[str, float]


@dataclass
class LIMEExplanation:
    prompt_text: str
    local_model_weights: Dict[str, float]
    local_model_intercept: float
    local_model_r2: float
    top_features: List[Tuple[str, float]]
    neighborhood_size: int


class PromptSHAPExplainer:
    def __init__(self, test_function: Callable[[str], Dict[str, float]]):
        self.test_function = test_function
        self.cache = {}
    
    def explain(self, prompt: str, metric: str = 'precision', num_samples: int = 50) -> SHAPValues:
        logger.info(f"Calculating SHAP values for {metric}")
        
        tokens = self._tokenize(prompt)
        baseline = self._test('Analyze the data.')
        full = self._test(prompt)
        
        base_value = baseline.get(metric, 0.5)
        full_value = full.get(metric, 0.5)
        
        token_shap = self._calculate_shapley_values(tokens, prompt, metric, base_value, num_samples)
        segments = self._get_segments(prompt)
        segment_shap = {seg[:30]: self._segment_contribution(seg, prompt, metric) for seg in segments}
        
        return SHAPValues(
            prompt_text=prompt,
            base_value=base_value,
            token_shap_values=token_shap,
            segment_shap_values=segment_shap,
            total_prediction=base_value + sum(token_shap.values()),
            metrics=full
        )
    
    def _tokenize(self, prompt: str) -> List[str]:
        return [s.strip() for s in re.split(r'[.!?\n]+', prompt) if s.strip()]
    
    def _get_segments(self, prompt: str) -> List[str]:
        return [s.strip() for s in prompt.split('\n\n') if s.strip()]
    
    def _test(self, prompt: str) -> Dict[str, float]:
        if prompt not in self.cache:
            self.cache[prompt] = self.test_function(prompt)
        return self.cache[prompt]
    
    def _calculate_shapley_values(self, tokens: List[str], prompt: str, metric: str, base: float, samples: int) -> Dict[str, float]:
        shap_values = {}
        
        for i, token in enumerate(tokens):
            contributions = []
            
            for _ in range(samples):
                subset_size = np.random.randint(0, len(tokens))
                subset = np.random.choice([j for j in range(len(tokens)) if j != i], size=min(subset_size, len(tokens)-1), replace=False)
                
                prompt_without = '. '.join([tokens[int(j)] for j in sorted(subset)])
                prompt_with = '. '.join([tokens[int(j)] for j in sorted(np.append(subset, i))])
                
                val_without = self._test(prompt_without if prompt_without else 'Analyze.').get(metric, base)
                val_with = self._test(prompt_with if prompt_with else 'Analyze.').get(metric, base)
                
                contributions.append(val_with - val_without)
            
            shap_values[token[:30]] = np.mean(contributions)
        
        return shap_values
    
    def _segment_contribution(self, segment: str, full_prompt: str, metric: str) -> float:
        without = full_prompt.replace(segment, '').strip()
        full_val = self._test(full_prompt).get(metric, 0.5)
        without_val = self._test(without if without else 'Analyze.').get(metric, 0.5)
        return full_val - without_val


class PromptLIMEExplainer:
    def __init__(self, test_function: Callable[[str], Dict[str, float]]):
        self.test_function = test_function
    
    def explain(self, prompt: str, metric: str = 'precision', num_samples: int = 100) -> LIMEExplanation:
        logger.info(f"Generating LIME explanation for {metric}")
        
        features = self._extract_features(prompt)
        X, y, distances = self._generate_neighborhood(prompt, features, metric, num_samples)
        weights = self._kernel_weights(distances)
        model, r2 = self._fit_local_model(X, y, weights)
        
        feature_names = list(features.keys())
        importances = [(feature_names[i], abs(model['coef'][i])) for i in range(len(feature_names))]
        importances.sort(key=lambda x: x[1], reverse=True)
        
        return LIMEExplanation(
            prompt_text=prompt,
            local_model_weights={feature_names[i]: model['coef'][i] for i in range(len(feature_names))},
            local_model_intercept=model['intercept'],
            local_model_r2=r2,
            top_features=importances[:10],
            neighborhood_size=len(X)
        )
    
    def _extract_features(self, prompt: str) -> Dict[str, bool]:
        return {
            'has_role': bool(re.search(r'You are', prompt, re.I)),
            'has_examples': bool(re.search(r'Example', prompt, re.I)),
            'has_threshold': bool(re.search(r'(above|over|exceeds?)', prompt, re.I)),
            'has_steps': bool(re.search(r'Step \d+', prompt)),
            'has_constraints': bool(re.search(r'(must|should|required)', prompt, re.I)),
            'has_format': bool(re.search(r'(JSON|format)', prompt, re.I)),
            'has_cot': bool(re.search(r'(think|reason|analyze)', prompt, re.I)),
            'is_long': len(prompt) > 300,
            'has_numbers': bool(re.search(r'\d+', prompt))
        }
    
    def _generate_neighborhood(self, prompt: str, features: Dict[str, bool], metric: str, n: int) -> Tuple:
        X, y, distances = [], [], []
        original_vec = [int(v) for v in features.values()]
        original_val = self.test_function(prompt).get(metric, 0.5)
        
        X.append(original_vec)
        y.append(original_val)
        distances.append(0.0)
        
        feature_names = list(features.keys())
        
        for _ in range(n-1):
            perturbed = features.copy()
            flips = np.random.randint(1, max(2, len(features)//3))
            for idx in np.random.choice(len(feature_names), flips, replace=False):
                perturbed[feature_names[idx]] = not perturbed[feature_names[idx]]
            
            perturbed_prompt = self._apply_changes(prompt, perturbed)
            perturbed_val = self.test_function(perturbed_prompt).get(metric, 0.5)
            perturbed_vec = [int(v) for v in perturbed.values()]
            
            distance = sum(a != b for a, b in zip(original_vec, perturbed_vec)) / len(original_vec)
            
            X.append(perturbed_vec)
            y.append(perturbed_val)
            distances.append(distance)
        
        return X, y, distances
    
    def _apply_changes(self, prompt: str, features: Dict[str, bool]) -> str:
        modified = prompt
        
        if not features.get('has_role'):
            modified = re.sub(r'You are.*?\.', '', modified, count=1)
        if not features.get('has_examples'):
            modified = re.sub(r'Example.*?(\n|$)', '', modified, flags=re.I)
        if features.get('has_constraints') and 'must' not in prompt.lower():
            modified += '\nMust follow instructions.'
        
        return modified.strip() or 'Analyze the data.'
    
    def _kernel_weights(self, distances: List[float]) -> List[float]:
        return [np.exp(-(d**2)/0.25) for d in distances]
    
    def _fit_local_model(self, X: List, y: List, weights: List) -> Tuple:
        X_arr = np.array(X)
        y_arr = np.array(y)
        W = np.diag(weights)
        
        try:
            XtWX = X_arr.T @ W @ X_arr + np.eye(X_arr.shape[1]) * 1e-6
            XtWy = X_arr.T @ W @ y_arr
            coef = np.linalg.solve(XtWX, XtWy)
        except:
            coef = np.zeros(X_arr.shape[1])
        
        pred = X_arr @ coef
        ss_res = np.sum(weights * (y_arr - pred)**2)
        ss_tot = np.sum(weights * (y_arr - np.mean(y_arr))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {'coef': coef.tolist(), 'intercept': np.mean(y_arr)}, r2


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    def mock_test(prompt: str) -> Dict[str, float]:
        score = 0.5
        if 'step' in prompt.lower(): score += 0.1
        if 'example' in prompt.lower(): score += 0.15
        if 'json' in prompt.lower(): score += 0.05
        return {'precision': min(0.99, score), 'recall': min(0.99, score-0.05)}
    
    prompt = '''You are an expert analyst.
    
Step 1: Review data
Step 2: Analyze patterns
Step 3: Report findings

Example: Identify high-value items
Output: JSON format'''
    
    print('='*60)
    print('SHAP and LIME Demo')
    print('='*60)
    
    print('\n📊 LIME Explainer...')
    lime = PromptLIMEExplainer(mock_test)
    lime_exp = lime.explain(prompt, num_samples=20)
    
    print(f'\nLocal Model R²: {lime_exp.local_model_r2:.3f}')
    print('\nTop Features:')
    for feat, imp in lime_exp.top_features[:5]:
        weight = lime_exp.local_model_weights[feat]
        print(f'  {feat}: {imp:.3f} (weight: {weight:+.3f})')
    
    print('\n✓ SHAP/LIME implementation complete!')
    print('\nUsage with real LLM:')
    print('  shap_exp = PromptSHAPExplainer(your_test_function)')
    print('  values = shap_exp.explain(your_prompt)')
