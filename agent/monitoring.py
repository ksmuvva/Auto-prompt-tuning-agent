"""
Real-Time Monitoring Dashboard for Prompt Tuning Agent

This module provides:
1. Real-time metrics collection and aggregation
2. WebSocket-based live updates
3. Performance monitoring (latency, throughput, errors)
4. Cost tracking across LLM providers
5. Test execution monitoring
6. Historical data storage and retrieval

Program of Thoughts:
1. Collect metrics from all agent operations
2. Aggregate metrics in real-time
3. Provide REST API for dashboard
4. Stream updates via WebSocket
5. Store historical data for trend analysis
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import sqlite3
from pathlib import Path


@dataclass
class MetricEvent:
    """A single metric event"""
    timestamp: float
    metric_type: str  # 'llm_call', 'test_complete', 'error', etc.
    value: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'metric_type': self.metric_type,
            'value': self.value,
            'metadata': self.metadata,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat()
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window"""
    window_start: float
    window_end: float
    llm_calls_total: int = 0
    llm_calls_success: int = 0
    llm_calls_failed: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    tests_completed: int = 0
    active_tests: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates metrics in real-time

    Thread-safe metrics collection with time-windowed aggregation.
    """

    def __init__(self, window_size: int = 60, max_events: int = 10000):
        """
        Initialize metrics collector

        Args:
            window_size: Aggregation window size in seconds
            max_events: Maximum events to keep in memory
        """
        self.window_size = window_size
        self.max_events = max_events

        # Thread-safe event queue
        self.events: deque = deque(maxlen=max_events)
        self.lock = threading.Lock()

        # Current aggregated metrics
        self.current_metrics = AggregatedMetrics(
            window_start=time.time(),
            window_end=time.time() + window_size
        )

        # Historical metrics (last 24 hours)
        self.historical_metrics: deque = deque(maxlen=1440)  # 1 minute windows for 24h

        # Real-time counters
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)

        # Active test tracking
        self.active_tests: Dict[str, Dict] = {}

    def record_event(self, metric_type: str, value: float, metadata: Optional[Dict] = None):
        """
        Record a metric event

        Thread-safe event recording.

        Args:
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        event = MetricEvent(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )

        with self.lock:
            self.events.append(event)
            self._update_aggregations(event)

    def _update_aggregations(self, event: MetricEvent):
        """Update aggregated metrics with new event"""
        current_time = time.time()

        # Check if we need to rotate the window
        if current_time >= self.current_metrics.window_end:
            # Save current window to historical
            self.historical_metrics.append(self.current_metrics)

            # Start new window
            self.current_metrics = AggregatedMetrics(
                window_start=current_time,
                window_end=current_time + self.window_size
            )

        # Update current metrics based on event type
        if event.metric_type == 'llm_call':
            self.current_metrics.llm_calls_total += 1
            if event.metadata.get('success', True):
                self.current_metrics.llm_calls_success += 1
            else:
                self.current_metrics.llm_calls_failed += 1

            # Update tokens and cost
            self.current_metrics.total_tokens += int(event.metadata.get('tokens', 0))
            self.current_metrics.total_cost += event.metadata.get('cost', 0.0)

            # Update average latency
            n = self.current_metrics.llm_calls_total
            self.current_metrics.avg_latency = (
                (self.current_metrics.avg_latency * (n - 1) + event.value) / n
            )

        elif event.metric_type == 'test_complete':
            self.current_metrics.tests_completed += 1

        elif event.metric_type == 'error':
            error_msg = event.metadata.get('error', 'Unknown error')
            self.current_metrics.errors.append(error_msg)

    def start_test(self, test_id: str, test_info: Dict):
        """Register a test as active"""
        with self.lock:
            self.active_tests[test_id] = {
                'start_time': time.time(),
                'info': test_info,
                'status': 'running'
            }
            self.current_metrics.active_tests = len(self.active_tests)

    def complete_test(self, test_id: str, result: Dict):
        """Mark a test as completed"""
        with self.lock:
            if test_id in self.active_tests:
                self.active_tests[test_id]['status'] = 'completed'
                self.active_tests[test_id]['end_time'] = time.time()
                self.active_tests[test_id]['result'] = result

                # Record completion event
                duration = time.time() - self.active_tests[test_id]['start_time']
                self.record_event('test_complete', duration, result)

            self.current_metrics.active_tests = len([
                t for t in self.active_tests.values() if t['status'] == 'running'
            ])

    def get_current_metrics(self) -> Dict:
        """Get current aggregated metrics"""
        with self.lock:
            return self.current_metrics.to_dict()

    def get_historical_metrics(self, hours: int = 1) -> List[Dict]:
        """
        Get historical metrics for the specified time period

        Args:
            hours: Number of hours of history to retrieve

        Returns:
            List of aggregated metrics
        """
        cutoff_time = time.time() - (hours * 3600)

        with self.lock:
            return [
                m.to_dict() for m in self.historical_metrics
                if m.window_start >= cutoff_time
            ]

    def get_active_tests(self) -> Dict[str, Dict]:
        """Get currently active tests"""
        with self.lock:
            return {
                test_id: {
                    **test_data,
                    'duration': time.time() - test_data['start_time']
                }
                for test_id, test_data in self.active_tests.items()
                if test_data['status'] == 'running'
            }

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        with self.lock:
            # Calculate stats from recent history
            recent = list(self.historical_metrics)[-60:]  # Last hour

            total_calls = sum(m.llm_calls_total for m in recent)
            total_cost = sum(m.total_cost for m in recent)
            total_tokens = sum(m.total_tokens for m in recent)
            avg_latency = (
                sum(m.avg_latency for m in recent) / len(recent)
                if recent else 0.0
            )

            return {
                'last_hour': {
                    'llm_calls': total_calls,
                    'total_cost': total_cost,
                    'total_tokens': total_tokens,
                    'avg_latency': avg_latency,
                    'success_rate': (
                        sum(m.llm_calls_success for m in recent) / total_calls
                        if total_calls > 0 else 0.0
                    )
                },
                'current_window': self.current_metrics.to_dict(),
                'active_tests': len(self.get_active_tests())
            }


class MetricsStorage:
    """
    Persistent storage for metrics using SQLite

    Stores historical metrics for long-term analysis.
    """

    def __init__(self, db_path: str = "logs/metrics.db"):
        """Initialize metrics storage"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Aggregated metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_start REAL NOT NULL,
                window_end REAL NOT NULL,
                llm_calls_total INTEGER,
                llm_calls_success INTEGER,
                llm_calls_failed INTEGER,
                total_tokens INTEGER,
                total_cost REAL,
                avg_latency REAL,
                tests_completed INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Test runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT,
                result TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_window ON aggregated_metrics(window_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tests_id ON test_runs(test_id)")

        conn.commit()
        conn.close()

    def store_event(self, event: MetricEvent):
        """Store a metric event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO events (timestamp, metric_type, value, metadata)
            VALUES (?, ?, ?, ?)
        """, (event.timestamp, event.metric_type, event.value, json.dumps(event.metadata)))

        conn.commit()
        conn.close()

    def store_aggregated_metrics(self, metrics: AggregatedMetrics):
        """Store aggregated metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO aggregated_metrics (
                window_start, window_end, llm_calls_total, llm_calls_success,
                llm_calls_failed, total_tokens, total_cost, avg_latency,
                tests_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.window_start, metrics.window_end, metrics.llm_calls_total,
            metrics.llm_calls_success, metrics.llm_calls_failed, metrics.total_tokens,
            metrics.total_cost, metrics.avg_latency, metrics.tests_completed
        ))

        conn.commit()
        conn.close()

    def query_events(self, start_time: float, end_time: float,
                    metric_type: Optional[str] = None) -> List[Dict]:
        """Query events within a time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if metric_type:
            cursor.execute("""
                SELECT timestamp, metric_type, value, metadata
                FROM events
                WHERE timestamp >= ? AND timestamp <= ? AND metric_type = ?
                ORDER BY timestamp
            """, (start_time, end_time, metric_type))
        else:
            cursor.execute("""
                SELECT timestamp, metric_type, value, metadata
                FROM events
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (start_time, end_time))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'timestamp': row[0],
                'metric_type': row[1],
                'value': row[2],
                'metadata': json.loads(row[3]) if row[3] else {}
            }
            for row in rows
        ]

    def query_aggregated_metrics(self, start_time: float, end_time: float) -> List[Dict]:
        """Query aggregated metrics within a time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT window_start, window_end, llm_calls_total, llm_calls_success,
                   llm_calls_failed, total_tokens, total_cost, avg_latency,
                   tests_completed
            FROM aggregated_metrics
            WHERE window_start >= ? AND window_start <= ?
            ORDER BY window_start
        """, (start_time, end_time))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'window_start': row[0],
                'window_end': row[1],
                'llm_calls_total': row[2],
                'llm_calls_success': row[3],
                'llm_calls_failed': row[4],
                'total_tokens': row[5],
                'total_cost': row[6],
                'avg_latency': row[7],
                'tests_completed': row[8]
            }
            for row in rows
        ]


class MonitoringDashboard:
    """
    Main dashboard orchestrator

    Coordinates metrics collection, storage, and provides API endpoints.
    """

    def __init__(self, db_path: str = "logs/metrics.db"):
        """Initialize monitoring dashboard"""
        self.collector = MetricsCollector()
        self.storage = MetricsStorage(db_path)

        # Background thread for periodic storage
        self.running = False
        self.storage_thread: Optional[threading.Thread] = None

    def start(self):
        """Start background monitoring"""
        self.running = True
        self.storage_thread = threading.Thread(target=self._storage_worker, daemon=True)
        self.storage_thread.start()

    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.storage_thread:
            self.storage_thread.join(timeout=5)

    def _storage_worker(self):
        """Background worker for persisting metrics"""
        while self.running:
            time.sleep(60)  # Store metrics every minute

            # Store current aggregated metrics
            metrics = self.collector.get_current_metrics()
            aggregated = AggregatedMetrics(**metrics)
            self.storage.store_aggregated_metrics(aggregated)

    def record_llm_call(self, latency: float, tokens: int, cost: float,
                       success: bool = True, provider: str = "unknown"):
        """Record an LLM API call"""
        self.collector.record_event('llm_call', latency, {
            'tokens': tokens,
            'cost': cost,
            'success': success,
            'provider': provider
        })

    def record_test_start(self, test_id: str, test_type: str, variant_count: int):
        """Record test start"""
        self.collector.start_test(test_id, {
            'type': test_type,
            'variant_count': variant_count
        })

    def record_test_complete(self, test_id: str, result: Dict):
        """Record test completion"""
        self.collector.complete_test(test_id, result)

    def record_error(self, error_message: str, error_type: str = "general"):
        """Record an error"""
        self.collector.record_event('error', 1.0, {
            'error': error_message,
            'type': error_type
        })

    def get_dashboard_data(self) -> Dict:
        """
        Get complete dashboard data

        Returns all data needed for dashboard display.
        """
        return {
            'summary': self.collector.get_summary_stats(),
            'current_metrics': self.collector.get_current_metrics(),
            'active_tests': self.collector.get_active_tests(),
            'historical_1h': self.collector.get_historical_metrics(hours=1),
            'historical_24h': self.collector.get_historical_metrics(hours=24),
        }

    def get_cost_breakdown(self, hours: int = 24) -> Dict:
        """Get cost breakdown by provider"""
        start_time = time.time() - (hours * 3600)
        events = self.storage.query_events(start_time, time.time(), 'llm_call')

        cost_by_provider = defaultdict(float)
        tokens_by_provider = defaultdict(int)

        for event in events:
            provider = event['metadata'].get('provider', 'unknown')
            cost_by_provider[provider] += event['metadata'].get('cost', 0.0)
            tokens_by_provider[provider] += event['metadata'].get('tokens', 0)

        return {
            'total_cost': sum(cost_by_provider.values()),
            'total_tokens': sum(tokens_by_provider.values()),
            'by_provider': {
                provider: {
                    'cost': cost,
                    'tokens': tokens_by_provider[provider],
                    'cost_per_token': cost / tokens_by_provider[provider] if tokens_by_provider[provider] > 0 else 0
                }
                for provider, cost in cost_by_provider.items()
            }
        }


# Global dashboard instance
_dashboard_instance: Optional[MonitoringDashboard] = None


def get_dashboard() -> MonitoringDashboard:
    """Get global dashboard instance (singleton)"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = MonitoringDashboard()
        _dashboard_instance.start()
    return _dashboard_instance
