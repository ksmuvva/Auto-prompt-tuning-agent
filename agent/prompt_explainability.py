"""
Prompt Explainability Module

Explains why prompts succeed/fail and what makes them effective.
Provides feature importance, attribution, and actionable improvement suggestions.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from collections import Counter
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptFeatureImportance:
    """Feature importance for prompt elements"""
    feature: str
    importance_score: float
    contribution_to_precision: float
    contribution_to_recall: float
    explanation: str
    examples: List[str] = field(default_factory=list)


@dataclass
class PromptAttribution:
    """Attribution of metrics to prompt segments"""
    segment: str
    segment_type: str
    precision_impact: float
    recall_impact: float
    confidence: float
    explanation: str


@dataclass
class PromptExplanation:
    """Complete explanation for a prompt's performance"""
    prompt_text: str
    metrics: Dict[str, float]
    feature_importances: List[PromptFeatureImportance]
    attributions: List[PromptAttribution]
    success_factors: List[str]
    failure_factors: List[str]
    improvement_suggestions: List[str]
    comparison_to_baseline: Dict[str, Any] = field(default_factory=dict)


class PromptExplainer:
    """Explains prompt performance and provides insights"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.prompt_features = self._define_prompt_features()
    
    def _define_prompt_features(self) -> Dict[str, str]:
        """Define analyzable features in prompts"""
        return {
            'has_role': r'You are (a|an) (.*?)(\.|\n)',
            'has_examples': r'(Example|For example|For instance)',
            'has_explicit_threshold': r'(above|over|exceeds?|greater than)\s*£?\s*\d+',
            'has_step_by_step': r'Step \d+|First|Second|Then|Finally',
            'has_constraints': r'(must|should|required|only|exclude)',
            'has_output_format': r'(JSON|format|structure|list)',
            'has_cot': r'(think|reason|consider|analyze)',
            'has_exclusions': r'(not|exclude|ignore|except)',
            'length': None,
            'complexity': None,
        }
    
    def explain_prompt_performance(
        self,
        prompt: str,
        metrics: Dict[str, Any],
        llm_response: str,
        ground_truth: Dict[str, Any],
        baseline_metrics: Dict[str, Any] = None
    ) -> PromptExplanation:
        """Generate comprehensive explanation for prompt performance"""
        
        features = self._extract_prompt_features(prompt)
        feature_importances = self._calculate_feature_importance(features, metrics, baseline_metrics)
        attributions = self._attribute_to_segments(prompt, metrics, llm_response)
        success_factors = self._identify_success_factors(prompt, metrics, features, attributions)
        failure_factors = self._identify_failure_factors(prompt, metrics, ground_truth, llm_response)
        suggestions = self._generate_improvement_suggestions(prompt, metrics, failure_factors, baseline_metrics)
        comparison = self._compare_to_baseline(metrics, baseline_metrics) if baseline_metrics else {}
        
        return PromptExplanation(
            prompt_text=prompt,
            metrics=metrics,
            feature_importances=feature_importances,
            attributions=attributions,
            success_factors=success_factors,
            failure_factors=failure_factors,
            improvement_suggestions=suggestions,
            comparison_to_baseline=comparison
        )
    
    def _extract_prompt_features(self, prompt: str) -> Dict[str, Any]:
        """Extract analyzable features from prompt"""
        features = {}
        
        for feature_name, pattern in self.prompt_features.items():
            if pattern is None:
                if feature_name == 'length':
                    features[feature_name] = len(prompt)
                elif feature_name == 'complexity':
                    sentences = prompt.split('.')
                    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                    features[feature_name] = avg_len
            else:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                features[feature_name] = len(matches) > 0
                features[f"{feature_name}_count"] = len(matches)
                if matches:
                    features[f"{feature_name}_examples"] = matches[:3]
        
        return features
    
    def _calculate_feature_importance(
        self, features: Dict[str, Any], metrics: Dict[str, Any], baseline_metrics: Dict[str, Any] = None
    ) -> List[PromptFeatureImportance]:
        """Calculate importance of each feature"""
        
        importances = []
        feature_impact = {
            'has_role': {'precision': 0.15, 'recall': 0.10, 'explanation': 'Role assignment helps LLM understand context'},
            'has_examples': {'precision': 0.25, 'recall': 0.20, 'explanation': 'Examples guide LLM output format'},
            'has_explicit_threshold': {'precision': 0.30, 'recall': 0.15, 'explanation': 'Explicit thresholds reduce ambiguity'},
            'has_step_by_step': {'precision': 0.20, 'recall': 0.25, 'explanation': 'Step-by-step improves systematic analysis'},
            'has_constraints': {'precision': 0.25, 'recall': -0.05, 'explanation': 'Constraints reduce false positives'},
            'has_output_format': {'precision': 0.10, 'recall': 0.10, 'explanation': 'Format specs improve parsability'},
            'has_exclusions': {'precision': 0.20, 'recall': -0.10, 'explanation': 'Exclusions prevent false positives'},
            'has_cot': {'precision': 0.15, 'recall': 0.15, 'explanation': 'Chain-of-thought improves quality'}
        }
        
        for feature_name, impacts in feature_impact.items():
            if features.get(feature_name):
                importance = PromptFeatureImportance(
                    feature=feature_name.replace('has_', ''),
                    importance_score=(abs(impacts['precision']) + abs(impacts['recall'])) / 2,
                    contribution_to_precision=impacts['precision'],
                    contribution_to_recall=impacts['recall'],
                    explanation=impacts['explanation'],
                    examples=features.get(f"{feature_name}_examples", [])
                )
                importances.append(importance)
        
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        return importances
    
    def _attribute_to_segments(self, prompt: str, metrics: Dict[str, Any], llm_response: str) -> List[PromptAttribution]:
        """Attribute performance to specific prompt segments"""
        segments = self._segment_prompt(prompt)
        attributions = []
        
        for segment, segment_type in segments:
            impact = self._estimate_segment_impact(segment, segment_type, metrics, llm_response)
            attribution = PromptAttribution(
                segment=segment[:100] + "..." if len(segment) > 100 else segment,
                segment_type=segment_type,
                precision_impact=impact['precision'],
                recall_impact=impact['recall'],
                confidence=impact['confidence'],
                explanation=impact['explanation']
            )
            attributions.append(attribution)
        
        return attributions
    
    def _segment_prompt(self, prompt: str) -> List[Tuple[str, str]]:
        """Segment prompt into analyzable parts"""
        segments = []
        parts = prompt.split('\n\n')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if re.search(r'You are (a|an)', part, re.IGNORECASE):
                segment_type = 'role'
            elif re.search(r'(Example|For example)', part, re.IGNORECASE):
                segment_type = 'example'
            elif re.search(r'(must|should|required|only)', part, re.IGNORECASE):
                segment_type = 'constraint'
            elif re.search(r'(format|structure|output)', part, re.IGNORECASE):
                segment_type = 'format'
            elif re.search(r'(analyze|identify|find|detect)', part, re.IGNORECASE):
                segment_type = 'task'
            else:
                segment_type = 'instruction'
            
            segments.append((part, segment_type))
        
        return segments
    
    def _estimate_segment_impact(self, segment: str, segment_type: str, metrics: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        """Estimate segment contribution to metrics"""
        impact = {
            'role': {'precision': 0.05, 'recall': 0.03, 'confidence': 0.6},
            'example': {'precision': 0.15, 'recall': 0.10, 'confidence': 0.7},
            'constraint': {'precision': 0.20, 'recall': -0.05, 'confidence': 0.8},
            'format': {'precision': 0.05, 'recall': 0.05, 'confidence': 0.9},
            'task': {'precision': 0.10, 'recall': 0.15, 'confidence': 0.7},
            'instruction': {'precision': 0.08, 'recall': 0.08, 'confidence': 0.5}
        }
        
        base_impact = impact.get(segment_type, {'precision': 0.05, 'recall': 0.05, 'confidence': 0.3})
        keywords = [w for w in segment.split() if len(w) > 4][:5]
        keyword_presence = sum(1 for kw in keywords if kw.lower() in llm_response.lower())
        keyword_ratio = keyword_presence / max(len(keywords), 1)
        base_impact['confidence'] *= (0.5 + 0.5 * keyword_ratio)
        base_impact['explanation'] = f"{segment_type.title()} segment with {keyword_ratio:.0%} keyword presence"
        
        return base_impact
    
    def _identify_success_factors(self, prompt: str, metrics: Dict[str, Any], features: Dict[str, Any], attributions: List[PromptAttribution]) -> List[str]:
        """Identify what made this prompt successful"""
        factors = []
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        if precision >= 0.95:
            factors.append(f"✓ Excellent precision ({precision:.1%}) - Strong constraints")
        if recall >= 0.95:
            factors.append(f"✓ Excellent recall ({recall:.1%}) - Comprehensive coverage")
        if features.get('has_explicit_threshold'):
            factors.append("✓ Explicit numerical threshold reduces ambiguity")
        if features.get('has_examples'):
            factors.append(f"✓ {features.get('has_examples_count', 0)} examples guide LLM")
        if features.get('has_step_by_step'):
            factors.append("✓ Step-by-step instructions ensure systematic analysis")
        
        return factors
    
    def _identify_failure_factors(self, prompt: str, metrics: Dict[str, Any], ground_truth: Dict[str, Any], llm_response: str) -> List[str]:
        """Identify what caused failures"""
        factors = []
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        fp_count = metrics.get('confusion_matrix', {}).get('false_positives', 0)
        fn_count = metrics.get('confusion_matrix', {}).get('false_negatives', 0)
        
        if precision < 0.98:
            factors.append(f"✗ {fp_count} false positives - Criteria too broad")
        if recall < 0.98:
            factors.append(f"✗ {fn_count} false negatives - Missing edge cases")
        if len(prompt) < 200:
            factors.append("✗ Prompt may be too brief")
        
        return factors
    
    def _generate_improvement_suggestions(self, prompt: str, metrics: Dict[str, Any], failure_factors: List[str], baseline_metrics: Dict[str, Any] = None) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        fp = metrics.get('confusion_matrix', {}).get('false_positives', 0)
        fn = metrics.get('confusion_matrix', {}).get('false_negatives', 0)
        
        if precision < 0.98 and fp > 0:
            suggestions.append(f"→ Add explicit exclusion criteria to reduce {fp} false positives")
        if recall < 0.98 and fn > 0:
            suggestions.append(f"→ Broaden criteria to capture {fn} missed cases")
        if 'step' not in prompt.lower():
            suggestions.append("→ Add step-by-step instructions")
        
        return suggestions
    
    def _compare_to_baseline(self, metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics to baseline"""
        comparison = {}
        
        for metric_name in ['precision', 'recall', 'accuracy', 'f1_score']:
            current = metrics.get(metric_name, 0)
            baseline = baseline_metrics.get(metric_name, 0)
            delta = current - baseline
            improvement = (delta / baseline * 100) if baseline > 0 else 0
            
            comparison[metric_name] = {
                'current': current,
                'baseline': baseline,
                'delta': delta,
                'improvement_pct': improvement,
                'better': delta > 0
            }
        
        return comparison
    
    def generate_explanation_report(self, explanation: PromptExplanation, output_format: str = 'markdown') -> str:
        """Generate human-readable explanation report"""
        if output_format == 'json':
            return json.dumps(asdict(explanation), indent=2, default=str)
        
        report = []
        report.append("# Prompt Performance Explanation\n")
        report.append("## 📊 Performance Metrics\n")
        report.append(f"- **Precision**: {explanation.metrics.get('precision', 0):.2%}")
        report.append(f"- **Recall**: {explanation.metrics.get('recall', 0):.2%}")
        report.append(f"- **Accuracy**: {explanation.metrics.get('accuracy', 0):.2%}")
        report.append(f"- **F1 Score**: {explanation.metrics.get('f1_score', 0):.2%}\n")
        
        if explanation.success_factors:
            report.append("## ✅ Success Factors\n")
            for factor in explanation.success_factors:
                report.append(f"{factor}")
            report.append("")
        
        if explanation.failure_factors:
            report.append("## ❌ Failure Factors\n")
            for factor in explanation.failure_factors:
                report.append(f"{factor}")
            report.append("")
        
        if explanation.feature_importances:
            report.append("## 🎯 Feature Importance\n")
            for fi in explanation.feature_importances[:5]:
                report.append(f"- **{fi.feature}** (Score: {fi.importance_score:.2f}): {fi.explanation}")
            report.append("")
        
        if explanation.improvement_suggestions:
            report.append("## 💡 Improvement Suggestions\n")
            for suggestion in explanation.improvement_suggestions:
                report.append(f"{suggestion}")
            report.append("")
        
        return "\n".join(report)


if __name__ == '__main__':
    import sys
    import io
    
    # Set UTF-8 encoding for output
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Demo
    explainer = PromptExplainer()
    
    demo_prompt = """You are an expert financial analyst.
    
Analyze transactions and identify high-value ones exceeding £250.

Step 1: Review amounts
Step 2: Compare against £250
Step 3: Flag transactions above threshold

Output: JSON array with transaction IDs"""
    
    demo_metrics = {
        'precision': 0.95,
        'recall': 0.92,
        'accuracy': 0.96,
        'f1_score': 0.935,
        'confusion_matrix': {'true_positives': 450, 'false_positives': 24, 'false_negatives': 40, 'true_negatives': 2400}
    }
    
    explanation = explainer.explain_prompt_performance(
        prompt=demo_prompt,
        metrics=demo_metrics,
        llm_response='["TXN001", "TXN002"]',
        ground_truth={'high_value_transactions': ['TXN001', 'TXN002']}
    )
    
    print(explainer.generate_explanation_report(explanation))