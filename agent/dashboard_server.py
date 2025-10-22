"""
Flask Web Server for Real-Time Monitoring Dashboard

Provides REST API and WebSocket endpoints for the monitoring dashboard.
"""

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
from .monitoring import get_dashboard


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


# HTML Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Tuning Agent - Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .header h1 { color: white; font-size: 2.5em; margin-bottom: 10px; }
        .header p { color: rgba(255, 255, 255, 0.9); font-size: 1.1em; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2a2a4e;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .metric-card h3 {
            color: #888;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #aaa;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .chart-container {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2a2a4e;
            margin-bottom: 20px;
        }
        .active-tests {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2a2a4e;
        }
        .test-item {
            background: #252544;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .status-running { background: #ffa500; color: #000; }
        .status-completed { background: #00c853; color: #000; }
        .status-failed { background: #ff5252; color: #fff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Prompt Tuning Agent Dashboard</h1>
        <p>Real-time monitoring and analytics</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <h3>LLM Calls (Last Hour)</h3>
            <div class="metric-value" id="llm-calls">0</div>
            <div class="metric-label">Total API calls</div>
        </div>
        <div class="metric-card">
            <h3>Success Rate</h3>
            <div class="metric-value" id="success-rate">0%</div>
            <div class="metric-label">API call success rate</div>
        </div>
        <div class="metric-card">
            <h3>Total Cost</h3>
            <div class="metric-value" id="total-cost">$0.00</div>
            <div class="metric-label">Last hour</div>
        </div>
        <div class="metric-card">
            <h3>Avg Latency</h3>
            <div class="metric-value" id="avg-latency">0ms</div>
            <div class="metric-label">Average response time</div>
        </div>
        <div class="metric-card">
            <h3>Active Tests</h3>
            <div class="metric-value" id="active-tests">0</div>
            <div class="metric-label">Currently running</div>
        </div>
        <div class="metric-card">
            <h3>Total Tokens</h3>
            <div class="metric-value" id="total-tokens">0</div>
            <div class="metric-label">Last hour</div>
        </div>
    </div>

    <div class="chart-container">
        <h3 style="margin-bottom: 15px; color: #e0e0e0;">Performance Over Time</h3>
        <canvas id="performanceChart"></canvas>
    </div>

    <div class="active-tests">
        <h3 style="margin-bottom: 15px; color: #e0e0e0;">Active Tests</h3>
        <div id="tests-list"></div>
    </div>

    <script>
        const socket = io();

        // Chart setup
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'LLM Calls',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Avg Latency (ms)',
                    data: [],
                    borderColor: '#ffa500',
                    backgroundColor: 'rgba(255, 165, 0, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { labels: { color: '#e0e0e0' } }
                },
                scales: {
                    x: { ticks: { color: '#888' }, grid: { color: '#2a2a4e' } },
                    y: { ticks: { color: '#888' }, grid: { color: '#2a2a4e' } },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        ticks: { color: '#888' },
                        grid: { display: false }
                    }
                }
            }
        });

        // Update metrics
        socket.on('metrics_update', function(data) {
            const summary = data.summary;
            const lastHour = summary.last_hour || {};

            document.getElementById('llm-calls').textContent = lastHour.llm_calls || 0;
            document.getElementById('success-rate').textContent =
                ((lastHour.success_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('total-cost').textContent =
                '$' + (lastHour.total_cost || 0).toFixed(4);
            document.getElementById('avg-latency').textContent =
                ((lastHour.avg_latency || 0) * 1000).toFixed(0) + 'ms';
            document.getElementById('active-tests').textContent =
                summary.active_tests || 0;
            document.getElementById('total-tokens').textContent =
                (lastHour.total_tokens || 0).toLocaleString();

            // Update chart
            const current = data.current_metrics;
            const time = new Date(current.window_start * 1000).toLocaleTimeString();

            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }

            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(current.llm_calls_total || 0);
            chart.data.datasets[1].data.push((current.avg_latency || 0) * 1000);
            chart.update();

            // Update active tests
            const testsList = document.getElementById('tests-list');
            const activeTests = data.active_tests || {};

            if (Object.keys(activeTests).length === 0) {
                testsList.innerHTML = '<p style="color: #888;">No active tests</p>';
            } else {
                testsList.innerHTML = Object.entries(activeTests).map(([id, test]) => `
                    <div class="test-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>${test.info.type}</strong>
                                <span class="status-badge status-${test.status}">${test.status}</span>
                            </div>
                            <div style="color: #888;">
                                Duration: ${test.duration.toFixed(1)}s
                            </div>
                        </div>
                        <div style="margin-top: 8px; color: #aaa; font-size: 0.9em;">
                            Variants: ${test.info.variant_count}
                        </div>
                    </div>
                `).join('');
            }
        });

        // Request initial data
        socket.emit('request_update');

        // Auto-refresh every 2 seconds
        setInterval(() => socket.emit('request_update'), 2000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    dashboard = get_dashboard()
    return jsonify(dashboard.get_dashboard_data())


@app.route('/api/cost-breakdown')
def get_cost_breakdown():
    """Get cost breakdown by provider"""
    hours = request.args.get('hours', default=24, type=int)
    dashboard = get_dashboard()
    return jsonify(dashboard.get_cost_breakdown(hours))


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('request_update')
def handle_update_request():
    """Handle metrics update request"""
    dashboard = get_dashboard()
    emit('metrics_update', dashboard.get_dashboard_data())


def broadcast_updates():
    """Background thread to broadcast metrics updates"""
    dashboard = get_dashboard()
    while True:
        time.sleep(2)  # Update every 2 seconds
        socketio.emit('metrics_update', dashboard.get_dashboard_data())


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """
    Run the dashboard server

    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    # Start background update thread
    update_thread = threading.Thread(target=broadcast_updates, daemon=True)
    update_thread.start()

    # Run server
    print(f"\nDashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_dashboard()
