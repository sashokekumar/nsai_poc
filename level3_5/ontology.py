# level3_5/ontology.py
"""
Ontology grounded in intents_base.csv:
- Entities and aliases come from phrases observed in the dataset (SRE intents)
- Symptoms reflect common investigative degradation/failure language in the dataset
- Time context patterns reflect dataset phrases (this quarter, last night, within the window, etc.)
"""

# Canonical intents aligned with dataset labels
INTENTS = ["summarization", "investigate", "execution", "out_of_scope"]

# -------------------------
# Entities (canonical)
# -------------------------
# Keep these canonical keys stable; grow aliases over time.
ENTITIES = {
        "general": {"type": "generic_scope"},
    # Kubernetes / Platform
    "horizontal_pod_autoscaler": {"type": "k8s_controller"},
    "kubernetes_node": {"type": "k8s_node"},
    "kubernetes_cluster": {"type": "k8s_cluster"},
    "pod": {"type": "k8s_workload"},
    "namespace": {"type": "k8s_namespace"},
    "resource_quota": {"type": "k8s_policy"},
    "metrics_server": {"type": "k8s_component"},
    "etcd": {"type": "k8s_datastore"},
    "flux": {"type": "gitops_controller"},

    # Networking / Traffic / Gateways
    "dns": {"type": "networking"},
    "network_policy": {"type": "networking_policy"},
    "load_balancer": {"type": "traffic_component"},
    "api_gateway": {"type": "traffic_component"},
    "reverse_proxy": {"type": "traffic_component"},
    "service_mesh": {"type": "traffic_component"},
    "circuit_breaker": {"type": "resilience_mechanism"},
    "rate_limiting": {"type": "resilience_mechanism"},
    "tls": {"type": "security_transport"},
    "certificate": {"type": "security_artifact"},

    # Application / Services
    "api": {"type": "application_interface"},
    "payment_service": {"type": "application_service"},
    "service": {"type": "application_service"},

    # Messaging / Streaming
    "message_queue": {"type": "messaging_component"},
    "kafka": {"type": "streaming_platform"},
    "consumer_group": {"type": "streaming_construct"},
    "kafka_topic": {"type": "streaming_construct"},
    "message_ack": {"type": "messaging_signal"},

    # Data / Storage
    "database": {"type": "data_store"},
    "read_replicas": {"type": "data_replication"},
    "connection_pool": {"type": "data_connectivity"},
    "cache": {"type": "data_cache"},
    "backup": {"type": "data_protection"},
    "storage": {"type": "storage"},
    "disk_io": {"type": "system_metric"},
    "volume_snapshot": {"type": "data_protection"},
    "snapshot_class": {"type": "data_protection"},

    # Ops Jobs / Pipelines / Releases
    "cron_job": {"type": "scheduler_job"},
    "batch_job": {"type": "batch_process"},
    "aggregation_job": {"type": "batch_process"},
    "ci_cd_pipeline": {"type": "delivery_pipeline"},
    "deployment": {"type": "delivery_event"},
    "blue_green_deployment": {"type": "delivery_strategy"},
    "container_image": {"type": "artifact"},
    "release_notes": {"type": "release_artifact"},
    "release_frequency": {"type": "release_metric"},

    # Observability / Reliability / Cost
    "observability_stack": {"type": "observability"},
    "logs": {"type": "observability_signal"},
    "metrics": {"type": "observability_signal"},
    "alerting": {"type": "observability_signal"},
    "distributed_tracing": {"type": "observability_signal"},
    "incident": {"type": "reliability_event"},
    "error_budget": {"type": "reliability_metric"},
    "availability_sla": {"type": "reliability_metric"},
    "scheduled_downtime": {"type": "reliability_event"},
    "load_test": {"type": "testing_event"},
    "cost_trends": {"type": "cost_metric"},
    "cost_anomalies": {"type": "cost_metric"},
    "health_check": {"type": "health_mechanism"},
}

# -------------------------
# Aliases observed in CSV
# -------------------------
# Each alias string is something that appears in your dataset text (or close surface form).
ENTITY_ALIASES = {
    "horizontal_pod_autoscaler": [
        "horizontal pod autoscaler",
        "hpa",
        "autoscaler",
    ],
    "kubernetes_node": [
        "kubernetes node",
        "cluster node",
        "node reporting not ready",
        "node is not ready",
        "node not ready",
        "node",
    ],
    "kubernetes_cluster": [
        "kubernetes cluster",
        "cluster",
    ],
    "pod": [
        "pod",
        "pod restart",
        "pod restart loop",
        "pod scheduling",
    ],
    "namespace": [
        "namespace",
    ],
    "resource_quota": [
        "resource quota",
        "quota consumption",
    ],
    "metrics_server": [
        "metrics server",
    ],
    "etcd": [
        "etcd",
        "etcd snapshot",
    ],
    "flux": [
        "flux",
        "flux reconciliation",
        "reconciliation interval",
    ],
    "dns": [
        "dns",
        "dns query",
        "dns change",
        "dns resolution",
        "dns resolution failure",
    ],
    "network_policy": [
        "network policy",
        "network policy audit",
    ],
    "load_balancer": [
        "load balancer",
    ],
    "api_gateway": [
        "api gateway",
    ],
    "reverse_proxy": [
        "reverse proxy",
        "proxy",
    ],
    "service_mesh": [
        "service mesh",
    ],
    "circuit_breaker": [
        "circuit breaker",
    ],
    "rate_limiting": [
        "rate limiting",
    ],
    "tls": [
        "tls",
        "tls handshake",
        "handshake failures",
        "tls handshake failures",
    ],
    "certificate": [
        "certificate",
        "certificate rotation",
    ],
    "api": [
        "api",
        "api service",
        "api endpoint",
    ],
    "payment_service": [
        "payment service",
    ],
    "service": [
        "service",
        "services",
        "service health",
        "service status",
        "across services",
    ],
    "message_queue": [
        "message queue",
        "queue consumers",
        "restart the message queue consumers",
        "queue",
    ],
    "message_ack": [
        "message acknowledgment",
        "acknowledgment",
        "ack is not being sent",
    ],
    "kafka": [
        "kafka",
        "kafka consumer",
        "slow kafka consumer",
    ],
    "consumer_group": [
        "consumer group",
        "consumer group commit",
    ],
    "kafka_topic": [
        "kafka topic",
        "topic",
    ],
    "database": [
        "database",
        "db",
        "query",
    ],
    "read_replicas": [
        "read replicas",
        "replicas",
    ],
    "connection_pool": [
        "connection pool",
    ],
    "cache": [
        "cache",
        "dns cache",
    ],
    "backup": [
        "backup",
        "backup window",
    ],
    "storage": [
        "storage",
        "storage growth rate",
        "storage utilization trends",
        "object storage",
    ],
    "disk_io": [
        "disk i/o",
        "disk i o",
        "disk io bottleneck",
        "disk i/o bottleneck",
        "i/o bottleneck",
        "disk bottleneck",
        "disk",
    ],
    "volume_snapshot": [
        "volume snapshot",
        "snapshot",
    ],
    "snapshot_class": [
        "snapshot class",
        "volume snapshot class",
    ],
    "cron_job": [
        "cron job",
        "cron jobs",
    ],
    "batch_job": [
        "batch job",
        "batch processing job",
    ],
    "aggregation_job": [
        "aggregation job",
        "producing empty output",
        "empty output",
    ],
    "ci_cd_pipeline": [
        "ci cd pipeline",
        "pipeline",
    ],
    "deployment": [
        "deployment",
    ],
    "blue_green_deployment": [
        "blue green deployment",
    ],
    "container_image": [
        "container image",
    ],
    "release_notes": [
        "release notes",
    ],
    "release_frequency": [
        "release frequency",
    ],
    "observability_stack": [
        "observability stack",
        "observability",
    ],
    "logs": [
        "log analysis",
        "logs",
        "log",
    ],
    "metrics": [
        "metrics",
        "key metrics",
    ],
    "alerting": [
        "alert trends",
        "alerting rule",
        "alerting",
        "alerts",
    ],
    "distributed_tracing": [
        "distributed tracing",
        "tracing",
    ],
    "incident": [
        "incidents",
        "platform incidents",
        "incident",
        "incident response",
    ],
    "error_budget": [
        "error budget",
    ],
    "availability_sla": [
        "availability sla",
        "sla",
    ],
    "scheduled_downtime": [
        "scheduled downtime",
        "downtime",
    ],
    "load_test": [
        "load test",
        "load test performance baseline",
    ],
    "cost_trends": [
        "cost trends",
        "cost",
        "cost by environment",
        "cost trends by environment",
    ],
    "cost_anomalies": [
        "cost anomalies",
        "infrastructure cost anomalies",
    ],
    "health_check": [
        "health check",
        "health checks",
        "health check interval",
    ],
}

# -------------------------
# Symptoms grounded in CSV investigative language
# -------------------------
# We normalize symptoms into canonical tokens.
SYMPTOM_ALIASES = {
    "failure": ["failure", "fail", "failing", "failures"],
    "error": ["error", "errors", "wrong response", "returning wrong"],
    "latency_high": ["latency", "response times", "increased response times"],
    "slow": ["slow", "slowness"],
    "timeout": ["timeout", "timing out"],
    "spike": ["spike"],
    "degraded": ["degraded", "throughput degraded", "degradation"],
    "crash": ["crash", "crashing"],
    "drop": ["drop", "dropped"],
    "leak": ["leak", "leaking"],
    "overflow": ["overflow"],
    "saturation": ["saturation", "connection saturation"],
    "bottleneck": ["bottleneck", "i/o bottleneck"],
    "not_ready": ["not ready"],
    "not_completing": ["not completing"],
    "restart_loop": ["restart loop", "restart loop on", "pod restart loop"],
    "empty_output": ["empty output", "producing empty output"],
    "tls_handshake_failure": ["tls handshake", "handshake failures", "tls handshake failures"],
}

# -------------------------
# Time context patterns grounded in CSV
# -------------------------
TIME_CONTEXT_PATTERNS = [
    ("last_night", ["last night"]),
    ("today", ["today"]),
    ("yesterday", ["yesterday"]),
    ("this_week", ["this week"]),
    ("last_week", ["last week"]),
    ("past_month", ["past month"]),
    ("this_quarter", ["this quarter"]),
    ("recent", ["recent"]),
    ("sla_window", ["within the window"]),
    ("last_24_hours", ["last 24 hours", "last 24"]),
    ("in_the_last", ["in the last"]),
    ("over_the_last", ["over the last"]),
    ("since", ["since"]),
    ("past", ["past "]),  # keep last: broad match
]