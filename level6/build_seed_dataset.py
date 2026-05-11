"""
level6/build_seed_dataset.py
----------------------------
Creates level6/data/level6_seed.csv — the failure-seeding dataset for Level 6.

The Level 5 model achieves ~96% intent accuracy on the existing 1,661-row
dataset, leaving too few natural failures for meaningful predicate-space
clustering. This script adds 300 harder utterances that probe known predicate
boundary cases, then derives the same 11 predicate columns used by Level 5.

Boundary cases targeted
-----------------------
  A. Multi-entity ambiguity        — utterances that plausibly match 2+ entity types
  B. Ambiguous SRE/non-SRE framing — SRE vocabulary in non-SRE context (false positives)
  C. Unknown entity + SRE intent   — valid SRE questions without a known entity type
  D. Runbook-absent execution      — execution requests with no procedural keywords
  E. Incident + execution blend    — utterances that mix incident and execution signals
  F. Metric + summarization blend  — metric queries that could be investigate or summarization
  G. Out-of-scope near-misses      — non-SRE utterances with accidental SRE vocabulary

The predicate derivation reuses the exact same keyword regex logic from
level5/build_dataset.py so predicate labels are consistent across levels.

Usage (from repo root):
    python -m level6.build_seed_dataset           # default: writes to level6/data/level6_seed.csv
    python -m level6.build_seed_dataset --audit   # print audit stats only, no write
    python -m level6.build_seed_dataset --out <path>
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
L5_CSV    = REPO_ROOT / "level5" / "data" / "level5_labeled.csv"
OUT_DIR   = REPO_ROOT / "level6" / "data"
OUT_CSV   = OUT_DIR / "level6_seed.csv"

# ---------------------------------------------------------------------------
# Predicate derivation — exact copy of level5/build_dataset.py logic
# (kept here to avoid a cross-module dependency at dataset-build time)
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "infrastructure", "service", "metric",
    "incident", "job", "pipeline", "unknown",
]

_SRE_KEYWORDS = re.compile(
    r"\b(pod|node|cluster|namespace|deployment|replica|service|mesh|gateway|"
    r"dns|latency|cpu|memory|disk|io|throughput|error.rate|sla|slo|alert|"
    r"incident|runbook|pipeline|job|kafka|queue|metric|log|trace|monitor|"
    r"observ|autoscal|health.check|circuit.break|rate.limit|tls|certificate|"
    r"backup|storage|database|cache|replica|load.balanc|proxy|api)\b",
    re.IGNORECASE,
)
_RUNBOOK_KEYWORDS = re.compile(
    r"\b(runbook|playbook|procedure|remediat|mitigat|rollback|restart|scale|"
    r"configure|set|inject|enable|disable|deploy|execute|run|trigger|apply)\b",
    re.IGNORECASE,
)
_INCIDENT_KEYWORDS = re.compile(
    r"\b(incident|outage|degradat|failure|alert|pagerduty|postmortem|"
    r"root.cause|rca|impact|down|crash|spike|anomal)\b",
    re.IGNORECASE,
)
_METRIC_KEYWORDS = re.compile(
    r"\b(metric|cpu|memory|latency|throughput|error.rate|p99|p95|percentile|"
    r"utiliz|trend|cost|sla|slo|dashboard|grafana|prometh|usage|saturat)\b",
    re.IGNORECASE,
)


def derive_predicates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for et in ENTITY_TYPES:
        out[f"is_{et}"] = (df["entity_type"] == et).astype(int)
    utt = df["utterance"].str.lower().fillna("")
    out["is_sre_domain"]     = utt.apply(lambda u: int(bool(_SRE_KEYWORDS.search(u))))
    out["has_runbook"]       = utt.apply(lambda u: int(bool(_RUNBOOK_KEYWORDS.search(u))))
    out["is_known_incident"] = utt.apply(lambda u: int(bool(_INCIDENT_KEYWORDS.search(u))))
    out["is_metric_query"]   = utt.apply(lambda u: int(bool(_METRIC_KEYWORDS.search(u))))
    return out


# ---------------------------------------------------------------------------
# Seed utterances — 300 rows across 7 boundary case categories
# Format: (utterance, intent, entity_type)
# entity_type must be one of ENTITY_TYPES
# intent must be one of: investigate, summarization, execution, out_of_scope
# ---------------------------------------------------------------------------

SEED_UTTERANCES = [
    # ------------------------------------------------------------------
    # A. Multi-entity ambiguity (43 rows)
    # Utterances that plausibly activate multiple is_* predicate heads.
    # Model must correctly resolve the dominant entity signal.
    # ------------------------------------------------------------------
    ("why is the service mesh causing high latency on the metric endpoint", "investigate", "service"),
    ("is the dns resolution failing for the database service", "investigate", "service"),
    ("why does the infrastructure alert keep firing for this metric", "investigate", "infrastructure"),
    ("show me the cpu metric for the payment service", "investigate", "service"),
    ("why is the cluster node reporting high memory on the kafka service", "investigate", "infrastructure"),
    ("is the pipeline job blocking the service deployment", "investigate", "pipeline"),
    ("what is the latency metric for the api gateway service", "investigate", "service"),
    ("why is the service pod crashing with a memory metric spike", "investigate", "service"),
    ("is the infrastructure dns affecting the service health check", "investigate", "infrastructure"),
    ("why is the job pod timing out under the current service load", "investigate", "job"),
    ("summarize the incident where the service latency exceeded the slo metric", "summarization", "incident"),
    ("give me a summary of the pipeline failure that impacted the service mesh", "summarization", "pipeline"),
    ("what happened when the cluster infrastructure incident caused metric degradation", "summarization", "incident"),
    ("recap the incident where a job failure elevated the error rate metric", "summarization", "incident"),
    ("summarize why the dns incident caused service latency spikes", "summarization", "incident"),
    ("describe the outage where infrastructure issues spiked the cpu metric", "summarization", "incident"),
    ("restart the service and then verify the latency metric", "execution", "service"),
    ("scale the infrastructure nodes and monitor the throughput metric", "execution", "infrastructure"),
    ("run the pipeline job and check the cpu metric afterward", "execution", "pipeline"),
    ("deploy the service and confirm the error rate metric normalizes", "execution", "service"),
    ("trigger the remediation runbook and monitor the service mesh metric", "execution", "service"),
    ("apply the rate limit config and check the api latency metric", "execution", "service"),
    ("enable monitoring for the infrastructure node and metric endpoint", "execution", "infrastructure"),
    ("why does the service pod show a metric anomaly during incident windows", "investigate", "service"),
    ("what is causing the infrastructure degradation that is spiking the slo metric", "investigate", "infrastructure"),
    ("is the job queue metric spiking because of the service mesh configuration", "investigate", "job"),
    ("why is the api gateway metric showing latency spikes during service rollout", "investigate", "service"),
    ("what is causing the pipeline metric to degrade after the infrastructure update", "investigate", "pipeline"),
    ("summarize the pipeline incident that degraded the job completion metric", "summarization", "pipeline"),
    ("recap the service outage that caused the infrastructure metric to spike", "summarization", "service"),
    ("describe the incident where the job failure was masked by the metric alert", "summarization", "incident"),
    ("summarize why the cluster node failure caused a service metric degradation", "summarization", "infrastructure"),
    ("what can you tell me about the infrastructure incident and the metric slo breach", "summarization", "incident"),
    ("restart the pipeline job and verify the service latency metric recovers", "execution", "pipeline"),
    ("scale up the infrastructure cluster and check the service throughput metric", "execution", "infrastructure"),
    ("deploy the updated service mesh config and monitor the latency metric", "execution", "service"),
    ("trigger the incident runbook for the service and track the metric normalization", "execution", "service"),
    ("configure the job retry policy and verify the pipeline metric stabilizes", "execution", "job"),
    ("why is the service reporting metric anomalies that look like an infrastructure issue", "investigate", "service"),
    ("is the job pod crash related to an infrastructure or a metric configuration problem", "investigate", "job"),
    ("what caused the api service to breach its slo metric during the infrastructure incident", "investigate", "service"),
    ("why is the pipeline degradation spiking the service latency metric", "investigate", "pipeline"),
    ("summarize the situation where the infrastructure failure affected the service slo metric", "summarization", "infrastructure"),

    # ------------------------------------------------------------------
    # B. Ambiguous SRE/non-SRE framing (44 rows)
    # Non-SRE contexts that contain SRE vocabulary — intended false positives
    # for is_sre_domain. Gold intent is out_of_scope.
    # ------------------------------------------------------------------
    ("what is the latency of light through glass", "out_of_scope", "unknown"),
    ("how do I monitor my blood pressure at home", "out_of_scope", "unknown"),
    ("what is the throughput capacity of a garden hose", "out_of_scope", "unknown"),
    ("why does the cpu in my laptop overheat when gaming", "out_of_scope", "unknown"),
    ("how do I deploy a tent at a campsite", "out_of_scope", "unknown"),
    ("what is the error rate on my tax return calculation", "out_of_scope", "unknown"),
    ("is my car's memory seat function working correctly", "out_of_scope", "unknown"),
    ("how do I scale a recipe for more servings", "out_of_scope", "unknown"),
    ("what is causing the alert sound on my smoke detector", "out_of_scope", "unknown"),
    ("should I restart my wifi router to fix connectivity", "out_of_scope", "unknown"),
    ("what is the sla for my gym membership contract", "out_of_scope", "unknown"),
    ("how do I configure the thermostat settings", "out_of_scope", "unknown"),
    ("what is the incident rate for car accidents in my city", "out_of_scope", "unknown"),
    ("how do I trigger an automatic savings transfer", "out_of_scope", "unknown"),
    ("what is the queue length at the post office today", "out_of_scope", "unknown"),
    ("can I enable automatic payments on my credit card", "out_of_scope", "unknown"),
    ("why is my gaming cluster lagging during peak hours", "out_of_scope", "unknown"),
    ("how do I set up a backup for my photos on icloud", "out_of_scope", "unknown"),
    ("what is the p95 salary for software engineers in seattle", "out_of_scope", "unknown"),
    ("how do I monitor stock performance over a rolling metric window", "out_of_scope", "unknown"),
    ("is the dns server used by my home router configurable", "out_of_scope", "unknown"),
    ("why is my phone memory full after the latest app update", "out_of_scope", "unknown"),
    ("how do I apply a rate limit on my internet plan", "out_of_scope", "unknown"),
    ("what is the throughput of my new printer per minute", "out_of_scope", "unknown"),
    ("can I set a memory limit on a browser tab", "out_of_scope", "unknown"),
    ("why does my laptop deployment of the software keep failing", "out_of_scope", "unknown"),
    ("how do I check the tls certificate on my personal website", "out_of_scope", "unknown"),
    ("what is the slo for a next-day delivery promise", "out_of_scope", "unknown"),
    ("how do I run a backup of my home nas storage", "out_of_scope", "unknown"),
    ("why is the replica of my podcast episode not available yet", "out_of_scope", "unknown"),
    ("how do I configure the gateway on my home network router", "out_of_scope", "unknown"),
    ("is there a circuit breaker on my home electrical panel that needs resetting", "out_of_scope", "unknown"),
    ("what is the log of my recent bank transactions", "out_of_scope", "unknown"),
    ("how do I trace the origin of a food delivery order", "out_of_scope", "unknown"),
    ("what is the cpu benchmark score of the new macbook", "out_of_scope", "unknown"),
    ("how do I configure proxy settings in firefox", "out_of_scope", "unknown"),
    ("what is the saturation level of colors in my monitor settings", "out_of_scope", "unknown"),
    ("how do I scale my freelance business revenue", "out_of_scope", "unknown"),
    ("why does my kafka subscription keep getting cancelled", "out_of_scope", "unknown"),
    ("can I monitor my sleep metric using a fitness tracker", "out_of_scope", "unknown"),
    ("how do I set a usage limit on my streaming subscription", "out_of_scope", "unknown"),
    ("what is the dashboard for tracking my investment portfolio", "out_of_scope", "unknown"),
    ("why is my node.js desktop app crashing on startup", "out_of_scope", "unknown"),
    ("can I deploy my website using github pages", "out_of_scope", "unknown"),

    # ------------------------------------------------------------------
    # C. Unknown entity + SRE intent (43 rows)
    # Legitimate SRE investigate/summarization questions where entity type
    # cannot be resolved to a known category. These probe the TYPE_C
    # constraint boundary (is_unknown + investigate/summarization).
    # ------------------------------------------------------------------
    ("why is the bloom filter showing unexpected collision rates", "investigate", "unknown"),
    ("what is causing the cache eviction policy to misbehave under load", "investigate", "unknown"),
    ("why is the consistent hashing ring unbalanced after the node addition", "investigate", "unknown"),
    ("what is causing the write-ahead log to grow unexpectedly", "investigate", "unknown"),
    ("why is the rate limiter not respecting the configured token bucket size", "investigate", "unknown"),
    ("what is the root cause of the shard rebalancing delays", "investigate", "unknown"),
    ("why is the circuit breaker state machine oscillating without a clear trigger", "investigate", "unknown"),
    ("what is causing the gossip protocol to diverge across regions", "investigate", "unknown"),
    ("why is the distributed lock timing out before the lease expires", "investigate", "unknown"),
    ("what caused the merkle tree reconciliation to fail during replication", "investigate", "unknown"),
    ("why is the fanout service degrading under high subscriber load", "investigate", "unknown"),
    ("what is causing the raft leader election to loop without converging", "investigate", "unknown"),
    ("why does the anti-entropy process keep finding inconsistencies after sync", "investigate", "unknown"),
    ("what is causing the backpressure mechanism to fail silently", "investigate", "unknown"),
    ("why is the saga orchestrator not completing compensation transactions", "investigate", "unknown"),
    ("what is causing the idempotency key collision in the payment processor", "investigate", "unknown"),
    ("why is the connection pool saturating under moderate request volume", "investigate", "unknown"),
    ("what caused the read repair to produce conflicting versions", "investigate", "unknown"),
    ("why is the health probe returning inconsistent results across replicas", "investigate", "unknown"),
    ("what is causing the sticky session affinity to break mid-flight", "investigate", "unknown"),
    ("why is the batch processor checkpointing at an unexpected offset", "investigate", "unknown"),
    ("what is causing the compaction process to stall the write path", "investigate", "unknown"),
    ("summarize the incident where the bloom filter caused a false positive cascade", "summarization", "unknown"),
    ("what happened when the distributed lock failure caused a split brain scenario", "summarization", "unknown"),
    ("give me a summary of the raft election storm that degraded the cluster", "summarization", "unknown"),
    ("describe the outage caused by the cache stampede during the flash sale", "summarization", "unknown"),
    ("recap the incident where the backpressure failure caused message loss", "summarization", "unknown"),
    ("summarize the postmortem of the saga compensation failure in the order service", "summarization", "unknown"),
    ("what happened during the gossip protocol divergence that caused regional inconsistency", "summarization", "unknown"),
    ("describe the incident where consistent hashing rebalancing caused latency spikes", "summarization", "unknown"),
    ("recap the write-ahead log overflow incident and its cascading effects", "summarization", "unknown"),
    ("summarize the shard rebalancing failure that caused query timeouts last week", "summarization", "unknown"),
    ("what can you tell me about the anti-entropy storm that hit production", "summarization", "unknown"),
    ("describe the incident where connection pool exhaustion cascaded to the api tier", "summarization", "unknown"),
    ("recap the read repair conflict that led to data inconsistency in the ledger", "summarization", "unknown"),
    ("what happened when the idempotency key collision caused duplicate payments", "summarization", "unknown"),
    ("describe the situation where the fanout service degradation caused notification delays", "summarization", "unknown"),
    ("summarize why the sticky session failure caused user session drops in prod", "summarization", "unknown"),
    ("what caused the batch checkpointing regression to restart all in-flight jobs", "summarization", "unknown"),
    ("recap the compaction stall incident that blocked writes for thirty minutes", "summarization", "unknown"),
    ("what happened during the circuit breaker oscillation that masked the root cause", "summarization", "unknown"),
    ("describe the saga timeout cascade that prevented order completion for an hour", "summarization", "unknown"),
    ("summarize the token bucket misconfiguration that throttled legitimate traffic", "summarization", "unknown"),

    # ------------------------------------------------------------------
    # D. Runbook-absent execution (43 rows)
    # Execution-intent utterances that do NOT contain runbook/procedural
    # keywords. These are hard for R2 (AND: has_runbook, is_sre_domain)
    # because has_runbook will be 0 despite the execution intent being correct.
    # ------------------------------------------------------------------
    ("bring the payment service back to a healthy state", "execution", "service"),
    ("get the cluster latency under the slo threshold", "execution", "infrastructure"),
    ("fix the broken dns resolution for the api gateway", "execution", "service"),
    ("clear the kafka consumer lag on the order topic", "execution", "pipeline"),
    ("stop the memory leak in the authentication service", "execution", "service"),
    ("resolve the disk saturation on the database node", "execution", "infrastructure"),
    ("correct the misconfigured rate limit on the checkout api", "execution", "service"),
    ("patch the tls certificate that is causing handshake failures", "execution", "service"),
    ("drain traffic from the degraded node in the cluster", "execution", "infrastructure"),
    ("flush the stale cache entries causing incorrect responses", "execution", "service"),
    ("remove the corrupted shard from the storage cluster", "execution", "infrastructure"),
    ("address the pod crash loop in the notification service", "execution", "service"),
    ("fix the broken health check endpoint on the load balancer", "execution", "infrastructure"),
    ("recover the failed pipeline job stuck in the queued state", "execution", "pipeline"),
    ("cut over traffic from the degraded data center region", "execution", "infrastructure"),
    ("force a leader re-election in the kafka broker cluster", "execution", "infrastructure"),
    ("evict the misbehaving node from the service mesh", "execution", "service"),
    ("terminate the stuck job that is blocking downstream pipeline stages", "execution", "job"),
    ("promote the standby database replica to primary", "execution", "infrastructure"),
    ("reduce the replica count on the overloaded storage service", "execution", "service"),
    ("drop the stale connection pool entries on the api service", "execution", "service"),
    ("compact the write-ahead log on the primary database node", "execution", "infrastructure"),
    ("revoke the leaked api credentials and rotate the service secret", "execution", "service"),
    ("update the dns records to point to the new load balancer ip", "execution", "infrastructure"),
    ("increase the timeout threshold for the downstream payment call", "execution", "service"),
    ("purge the dead letter queue messages older than seven days", "execution", "pipeline"),
    ("bring the canary deployment to full traffic immediately", "execution", "service"),
    ("redirect the ingress traffic away from the saturated cluster", "execution", "infrastructure"),
    ("suspend the batch job that is saturating the database cpu", "execution", "job"),
    ("force a full resync of the replication lag on the secondary node", "execution", "infrastructure"),
    ("reset the circuit breaker state on the payment gateway service", "execution", "service"),
    ("hard-reset the stuck kafka partition assignment on the broker", "execution", "infrastructure"),
    ("mark the failed pipeline task as skipped and unblock downstream jobs", "execution", "pipeline"),
    ("move the high-cpu job to the dedicated batch node pool", "execution", "job"),
    ("expand the disk volume on the logging infrastructure node", "execution", "infrastructure"),
    ("clear the expired sessions from the authentication service cache", "execution", "service"),
    ("switch the metric scrape endpoint to the new prometheus target", "execution", "metric"),
    ("stop accepting new connections on the overloaded api gateway", "execution", "service"),
    ("replace the failing tls certificate on the ingress controller", "execution", "infrastructure"),
    ("halt the runaway autoscaler that is provisioning excess nodes", "execution", "infrastructure"),
    ("correct the pod affinity rules that are packing jobs onto a single node", "execution", "job"),
    ("roll back the faulty deployment that introduced the latency regression", "execution", "service"),
    ("isolate the noisy-neighbor container that is starving the service cpu", "execution", "infrastructure"),

    # ------------------------------------------------------------------
    # E. Incident + execution blend (43 rows)
    # Utterances that mix incident signals with execution intent.
    # is_known_incident fires (→ R1 investigte / R3 summarization) but
    # the gold intent is execution. Tests whether rule layer overrides correctly.
    # ------------------------------------------------------------------
    ("resolve the active incident by restarting the payment service", "execution", "incident"),
    ("during the current outage scale up the database cluster nodes", "execution", "incident"),
    ("mitigate the ongoing degradation by rolling back the api service deployment", "execution", "incident"),
    ("while the incident is open drain traffic from the impacted region", "execution", "incident"),
    ("apply the emergency runbook to contain the active service outage", "execution", "incident"),
    ("trigger the incident remediation by restarting the failed kafka broker", "execution", "incident"),
    ("execute the rollback procedure for the deployment causing the current incident", "execution", "incident"),
    ("resolve the alert by scaling the cluster during the active incident", "execution", "incident"),
    ("cut over to the backup region to contain the ongoing degradation", "execution", "incident"),
    ("close the incident by enabling the circuit breaker on the payment gateway", "execution", "incident"),
    ("execute the incident runbook step to flush the corrupted cache", "execution", "incident"),
    ("during the active pagerduty alert restart the authentication service", "execution", "incident"),
    ("apply rate limiting to contain the traffic spike causing the incident", "execution", "incident"),
    ("while the outage is active patch the tls certificate causing handshake failures", "execution", "incident"),
    ("run the postmortem action to disable the faulty feature flag", "execution", "incident"),
    ("force failover the degraded database primary during the active incident", "execution", "incident"),
    ("promote the replica to primary to resolve the current database outage", "execution", "incident"),
    ("remediate the incident by terminating the stuck pipeline job", "execution", "incident"),
    ("contain the anomaly by scaling down the noisy job pool", "execution", "incident"),
    ("trigger a deployment rollback to fix the crash loop causing the current alert", "execution", "incident"),
    ("fix the root cause of the current incident by updating the dns record", "execution", "incident"),
    ("while the alert is firing suspend the batch job spiking the cpu", "execution", "incident"),
    ("resolve the degradation by adjusting the rate limiter during the incident window", "execution", "incident"),
    ("execute the mitigation step in the active incident runbook to stop the memory leak", "execution", "incident"),
    ("bring the incident to resolution by draining the saturated queue", "execution", "incident"),
    ("apply the incident fix by increasing the connection pool on the api gateway", "execution", "incident"),
    ("during the active outage evict the misbehaving node from the service mesh", "execution", "incident"),
    ("remediate the alert by promoting the standby replica to primary", "execution", "incident"),
    ("while the incident is active compact the log on the database primary", "execution", "incident"),
    ("resolve the service crash by reverting the misconfigured infrastructure setting", "execution", "incident"),
    ("clear the alert by resetting the circuit breaker in the incident playbook", "execution", "incident"),
    ("apply the emergency scale-out to resolve the active slo breach", "execution", "incident"),
    ("take the degraded node offline to contain the incident impact", "execution", "incident"),
    ("execute the recovery procedure for the ongoing kafka broker failure", "execution", "incident"),
    ("force a leader election to fix the stuck kafka partition during the incident", "execution", "incident"),
    ("apply the incident mitigation by redirecting traffic to the healthy region", "execution", "incident"),
    ("during the active degradation purge the corrupted queue messages", "execution", "incident"),
    ("remediate the ongoing outage by rolling back the failed infrastructure change", "execution", "incident"),
    ("resolve the active alert by clearing expired sessions on the auth service", "execution", "incident"),
    ("trigger the circuit breaker reset as part of the ongoing incident response", "execution", "incident"),
    ("fix the incident by correcting the pod affinity misconfiguration", "execution", "incident"),
    ("apply the hotfix deployment to resolve the service crash in the active incident", "execution", "incident"),
    ("use the incident runbook to drain traffic from the overloaded api cluster", "execution", "incident"),

    # ------------------------------------------------------------------
    # F. Metric + summarization blend (43 rows)
    # Metric queries that could resolve as investigate OR summarization.
    # Tests whether the rule layer correctly distinguishes historical recap
    # (summarization) from diagnostic question (investigate).
    # ------------------------------------------------------------------
    ("what were the p99 latency numbers during last week's incident", "summarization", "metric"),
    ("give me a summary of the error rate trend over the past month", "summarization", "metric"),
    ("what was the cpu utilization profile during the outage window", "summarization", "metric"),
    ("recap the throughput degradation metrics from last tuesday's event", "summarization", "metric"),
    ("summarize the slo compliance metrics for the payment service in q1", "summarization", "metric"),
    ("what did the latency metrics look like during the deployment rollout", "summarization", "metric"),
    ("give me a historical view of the memory utilization trend on the cluster", "summarization", "metric"),
    ("what were the error rate peaks recorded during the incident postmortem", "summarization", "metric"),
    ("summarize the metric dashboard observations from the production incident", "summarization", "metric"),
    ("describe the cpu saturation pattern observed during last quarter", "summarization", "metric"),
    ("what were the key metric signals before the service degradation started", "summarization", "metric"),
    ("give me a recap of the sla breach metrics from the infrastructure incident", "summarization", "metric"),
    ("what was the throughput profile during the peak traffic event last month", "summarization", "metric"),
    ("summarize the prometheus metrics captured during the raft election storm", "summarization", "metric"),
    ("what did the grafana dashboard show during the outage last friday", "summarization", "metric"),
    ("recap the p95 latency metrics logged during the cache stampede event", "summarization", "metric"),
    ("what were the saturation metrics for the kafka cluster during the incident", "summarization", "metric"),
    ("summarize the metric trend that preceded the connection pool exhaustion", "summarization", "metric"),
    ("describe the usage pattern that led to the slo breach in the api service", "summarization", "metric"),
    ("what latency metrics were observed across regions during the dns incident", "summarization", "metric"),
    ("why is the p99 latency spiking on the checkout service right now", "investigate", "metric"),
    ("what is causing the cpu utilization to exceed the threshold on the api node", "investigate", "metric"),
    ("why is the error rate climbing on the payment service metric endpoint", "investigate", "metric"),
    ("what is driving the memory usage spike on the database cluster", "investigate", "metric"),
    ("why does the throughput metric show a cliff drop every six hours", "investigate", "metric"),
    ("what is causing the slo metric to degrade only during business hours", "investigate", "metric"),
    ("why is the prometh scrape failing for the service latency metric", "investigate", "metric"),
    ("what is the root cause of the saturation reading on the kafka broker metric", "investigate", "metric"),
    ("why is the grafana dashboard showing inconsistent error rate metrics", "investigate", "metric"),
    ("what is causing the p95 metric to spike after every deployment", "investigate", "metric"),
    ("why does the cost metric increase sharply after the autoscaler fires", "investigate", "metric"),
    ("what is driving the utilization metric anomaly on the storage node", "investigate", "metric"),
    ("why is the sla metric dropping only on the european region endpoint", "investigate", "metric"),
    ("what is causing the disk io metric to saturate before the backup job runs", "investigate", "metric"),
    ("why is the dashboard metric showing a memory leak pattern on the api service", "investigate", "metric"),
    ("what is the cpu metric trend telling us about the current cluster health", "investigate", "metric"),
    ("why is the throughput metric diverging between the primary and replica nodes", "investigate", "metric"),
    ("what is causing the latency percentile metric to widen during peak load", "investigate", "metric"),
    ("why does the cost metric spike correlate with the deployment pipeline metric", "investigate", "metric"),
    ("what is driving the error rate metric degradation after the config change", "investigate", "metric"),
    ("why is the saturation metric not triggering the expected autoscaler response", "investigate", "metric"),
    ("what does the prometh metric show about the service connection pool health", "investigate", "metric"),
    ("why is the slo metric breaching only when the batch job metric peaks", "investigate", "metric"),

    # ------------------------------------------------------------------
    # G. Out-of-scope near-misses (41 rows)
    # Non-SRE utterances that are very close to SRE phrasing but are
    # genuinely out-of-scope. Harder than category B because they use
    # more specific SRE structural vocabulary (cluster, pod, replica, etc.)
    # in non-infrastructure contexts.
    # ------------------------------------------------------------------
    ("why is my kubernetes homework assignment failing the autograder", "out_of_scope", "unknown"),
    ("how do I set up a home lab cluster with raspberry pi nodes", "out_of_scope", "unknown"),
    ("can I run a prometheus tutorial locally without docker", "out_of_scope", "unknown"),
    ("what is the best runbook for learning terraform from scratch", "out_of_scope", "unknown"),
    ("how do I write a blog post about incident management best practices", "out_of_scope", "unknown"),
    ("what certification do I need to become an sre", "out_of_scope", "unknown"),
    ("is grafana free to use for personal dashboards", "out_of_scope", "unknown"),
    ("how do I set up prometheus alerting on my home server", "out_of_scope", "unknown"),
    ("what is the best book about distributed systems for beginners", "out_of_scope", "unknown"),
    ("can I use kafka for a personal messaging app", "out_of_scope", "unknown"),
    ("how do I create a replica of my podcast for a backup channel", "out_of_scope", "unknown"),
    ("what is the pod size limit for checked luggage on this airline", "out_of_scope", "unknown"),
    ("how do I scale my etsy shop to handle more orders", "out_of_scope", "unknown"),
    ("what is the latency between sydney and new york on a fiber cable", "out_of_scope", "unknown"),
    ("how do I monitor my sleep patterns using a wearable", "out_of_scope", "unknown"),
    ("what is an slo and how is it different from a kpi in marketing", "out_of_scope", "unknown"),
    ("can I deploy a static website on an s3 bucket for free", "out_of_scope", "unknown"),
    ("how do I configure the dns on my raspberry pi for local resolution", "out_of_scope", "unknown"),
    ("what are the best practices for api design in a microservices tutorial", "out_of_scope", "unknown"),
    ("how do I run a postmortem retro for my startup team", "out_of_scope", "unknown"),
    ("what is the difference between a runbook and an sop in a hospital", "out_of_scope", "unknown"),
    ("can I use kubernetes to orchestrate my machine learning training jobs", "out_of_scope", "unknown"),
    ("how do I set up rate limiting on my personal api built with fastapi", "out_of_scope", "unknown"),
    ("what is the best way to learn about circuit breaker patterns for interviews", "out_of_scope", "unknown"),
    ("how do I add a health check endpoint to my django app", "out_of_scope", "unknown"),
    ("what is the node version I should use for my react project", "out_of_scope", "unknown"),
    ("how do I restart my raspberry pi node after a power failure", "out_of_scope", "unknown"),
    ("can I use grafana cloud for a school data visualization project", "out_of_scope", "unknown"),
    ("what is the error rate on my university exam scores this semester", "out_of_scope", "unknown"),
    ("how do I deploy a machine learning model using flask on my laptop", "out_of_scope", "unknown"),
    ("what is the memory limit for a google colab notebook session", "out_of_scope", "unknown"),
    ("how do I use prometheus metrics in a personal python project", "out_of_scope", "unknown"),
    ("can I scale a discord bot to handle multiple servers at once", "out_of_scope", "unknown"),
    ("what is the throughput of a consumer graphics card for llm inference", "out_of_scope", "unknown"),
    ("how do I set up a load balancer for my personal minecraft server", "out_of_scope", "unknown"),
    ("can a circuit breaker pattern be used in a mobile app", "out_of_scope", "unknown"),
    ("what is the sla difference between google drive and dropbox for consumers", "out_of_scope", "unknown"),
    ("how do I configure tls on my personal blog hosted on nginx", "out_of_scope", "unknown"),
    ("what is the latency of a satellite internet connection compared to fiber", "out_of_scope", "unknown"),
    ("can I use kafka streams for a real-time game leaderboard project", "out_of_scope", "unknown"),
    ("how do I monitor the cpu usage of my gaming pc during a stream", "out_of_scope", "unknown"),
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(audit_only: bool = False, out_path: Path = OUT_CSV) -> pd.DataFrame:
    assert len(SEED_UTTERANCES) == 300, (
        f"Expected exactly 300 seed utterances, got {len(SEED_UTTERANCES)}. "
        "Adjust SEED_UTTERANCES list."
    )

    # Load existing Level 5 dataset
    base_df = pd.read_csv(L5_CSV)
    print(f"[build_seed_dataset] Base L5 dataset : {len(base_df)} rows")

    # Build seed rows
    seed_df = pd.DataFrame(
        SEED_UTTERANCES, columns=["utterance", "intent", "entity_type"]
    )
    seed_df["domain_valid"] = (seed_df["intent"] != "out_of_scope").astype(int)
    seed_df["source"] = "level6_seed"

    # Tag base rows
    base_df = base_df.copy()
    base_df["source"] = "level5_base"

    # Derive predicates for seed rows
    seed_df = derive_predicates(seed_df)

    # Ensure column alignment
    pred_cols = [f"is_{et}" for et in ENTITY_TYPES] + [
        "is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query"
    ]
    col_order = ["utterance", "intent", "entity_type", "domain_valid", "source"] + pred_cols
    for col in col_order:
        if col not in base_df.columns:
            base_df[col] = None
        if col not in seed_df.columns:
            seed_df[col] = None

    combined = pd.concat(
        [base_df[col_order], seed_df[col_order]], ignore_index=True
    )

    print(f"[build_seed_dataset] Seed rows added  : {len(seed_df)}")
    print(f"[build_seed_dataset] Combined total   : {len(combined)}")

    _audit(combined, seed_df)

    if not audit_only:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"[build_seed_dataset] Written to       : {out_path}")

    return combined


def _audit(combined: pd.DataFrame, seed_df: pd.DataFrame):
    print()
    print("=" * 60)
    print("  Level 6 Seed Dataset Audit")
    print("=" * 60)
    print(f"Total rows   : {len(combined)}")
    print(f"  L5 base    : {(combined['source'] == 'level5_base').sum()}")
    print(f"  L6 seed    : {(combined['source'] == 'level6_seed').sum()}")
    print()

    print("Intent distribution (seed rows only):")
    print(seed_df["intent"].value_counts().to_string())
    print()

    print("Entity type distribution (seed rows only):")
    print(seed_df["entity_type"].value_counts().to_string())
    print()

    pred_cols = [f"is_{et}" for et in ENTITY_TYPES] + [
        "is_sre_domain", "has_runbook", "is_known_incident", "is_metric_query"
    ]
    print("Predicate coverage — seed rows (% = 1):")
    for col in pred_cols:
        pct = seed_df[col].mean() * 100
        print(f"  {col:<22}  {int(seed_df[col].sum()):>4} / {len(seed_df)}  ({pct:5.1f}%)")
    print()

    print("Null check (combined):",
          combined.isnull().sum()[combined.isnull().sum() > 0].to_dict() or "none")
    print()

    # Category coverage
    cats = {
        "A multi-entity ambiguity":      43,
        "B ambiguous SRE/non-SRE":       44,
        "C unknown entity + SRE intent": 43,
        "D runbook-absent execution":    43,
        "E incident + execution blend":  43,
        "F metric + summarization blend":43,
        "G out-of-scope near-misses":    41,
    }
    total = sum(cats.values())
    print(f"Boundary case category breakdown (target total: {total}):")
    for cat, count in cats.items():
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {total}")
    assert total == 300

    print()
    print("VERDICT: level6_seed.csv READY" if combined.isnull().sum().sum() == 0
          else "WARNING: nulls found in combined dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Level 6 seed dataset.")
    parser.add_argument("--audit", action="store_true",
                        help="Print audit stats only, do not write CSV.")
    parser.add_argument("--out", type=str, default=str(OUT_CSV),
                        help="Output CSV path (default: level6/data/level6_seed.csv).")
    args = parser.parse_args()
    build(audit_only=args.audit, out_path=Path(args.out))
