from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNTER = Counter(
    "driftwatch_requests_total",
    "Total proxied requests",
    ["model", "stream"],
)

LATENCY_HISTOGRAM = Histogram(
    "driftwatch_latency_ms",
    "Request latency in ms",
    buckets=[50, 100, 250, 500, 1000, 2000, 5000],
)

DRIFT_ALERT_COUNTER = Counter(
    "driftwatch_drift_alerts_total",
    "Total drift alerts fired",
)

QUALITY_SCORE_GAUGE = Gauge(
    "driftwatch_quality_score_latest",
    "Latest quality score from evaluator",
)

GOLDEN_SET_SIZE = Gauge(
    "driftwatch_golden_set_size",
    "Current number of golden set entries",
)
