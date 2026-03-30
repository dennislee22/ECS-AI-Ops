<h1><img src="web/static/chatbot-icon.svg" width="30" height="30"> <img src="web/static/k8s-logo.svg" width="30" height="30"> Cloudera ECS AI Ops Chatbot <img src="web/static/rancher-logo.svg" width="50" height="50"> <img src="web/static/longhorn-logo.svg" width="30" height="30"> </h1>

- An Agentic AI assistant for your live **Cloudera ECS (Embedded Container Service)** cluster (highly curated for Rancher K8s distro and Longhorn storage) — ask it anything about your cluster in plain English and get informative answers.
- No need to memorize `kubectl` commands or deal with tedious YAML/JSON; only a basic understanding of Kubernetes is required.
- Search for known ECS/Longhorn incidents, best practices, and troubleshooting guides via RAG (LanceDB).
- Air-gap–friendly deployment, it runs fully within isolated environments with no external internet access. The LLM can be hosted on GPU or CPU, no code changes required. 

<img src="https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/ecs-ai-ops-dashboard1.png" alt="Dashboard"/>

**What can it do?**

- 🔍 **Check cluster health**: ask about the state of pod, node, namespace, storage (longhorn), ingress, etc.
- 🔍 **Resource Monitoring**: monitor resources like CPU, memory, disk I/O, persistentVolume (PV) usage, ingress traffics, etc.
- 🔍 **Configuration and Settings**: retrieve configurations for Longhorn, secret, configMaps, and others.
- 🔍 **Troubleshooting**: help diagnose issues with pods, services, and other cluster components.
- 🔍 **Query databases directly**: run read-only SQL inside DB pods (MySQL, PostgreSQL) straight from the chat interface.

ECS AI Ops Chatbot is powered by:

- 🤖 **Self-hosted local LLM**: runs entirely on-premise with no external API calls or internet dependency. Use **[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** (HuggingFace Transformers) on GPU. When no GPU is detected at startup, the application automatically falls back to CPU inference. `Qwen3-8B` is selected for its robust native tool-calling capabilities while remaining lightweight, helping conserve tokens. Alternatively, use **[Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M** (llama-cpp-python) on CPU.
- 🔁 **LangGraph agentic loop**: a ReAct agent that autonomously selects the right Kubernetes tools, executes them, observes the results, and chains further calls when needed before synthesising a final answer.
- 📚 **LanceDB RAG**: cross-references live cluster data against your own runbooks, known-issue docs, and SOPs ingested locally into a LanceDB vector store.

<img src="https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/ecs-ai-arch.gif" alt="arch-gif"/>

---
> ⚠️ **Note:** While the tooling is built on the Kubernetes Python SDK, the system prompt, tool selection logic, and multi-hop reasoning chains are highly curated for ECS and may not work correctly on other Kubernetes distributions with different storage or networking subsystems.
---

## Table of Contents

- [Project Structure](#project-structure)
- [Stack](#stack)
- [Kubernetes Tools](#kubernetes-tools)
- [Example Queries](#example-queries)
- [Quick Start](#quick-start)
- [Quick Start in Openshift](#quick-start-in-openshift)
- [REST API](#rest-api)
- [Hardware Sizing](#hardware-sizing)
- [Security Notes](#security-notes)
- [Demo](#demo)
- [Contact](#contact)

---

## Project Structure

```
├── app.py                    # FastAPI server + LangGraph agent
├── ocp_buildapp.py           # Build and run in Openshift
├── .s2i/bin                  # This is needed only if you run this in Openshift
├── env                       # Optional to declare specific environment variable(s)
├── config/
│   ├── __init__.py
│   ├── config.py             # Configuration file
│   ├── settings.json.        # Generated automatically by the chatbot according to ⚙️ Settings in the Chatbot UI
│   ├── system_prompt.txt     # ECS AI Ops Chatbot system prompt
│   └── kb_prompt.txt         # ECS KB Bot system prompt
├── agent/
│   ├── __init__.py
│   ├── routing.py            # Namespace resolution + emergency fallback routing
│   └── bypass.py             # LLM synthesis bypass for simple list queries
├── tools/
│   ├── __init__.py
│   ├── tools_k8s.py          # K8s tools
│   └── tools_metadata.py     # K8s tools metadata
├── web/
│   ├── index.html            # Main dashboard (served by FastAPI)
│   └── static/
│       ├── k8s-logo.svg
│       ├── rancher-logo.svg
│       ├── longhorn-logo.svg
│       ├── ibm-plex.css
│       └── marked.min.js
├── rag/
│   ├── __init__.py
│   ├── tool_index.py         # Index K8s tools for LanceDB to speed up tool selection by LLM
│   ├── ingest.txt            # Ingest document
│   ├── retrieve.py           # Retrieve RAG data
│   ├── store.py              # Store RAG
│   └── sample     
│       └── ECS_KB.xls        # Sample xls
├── requirements.txt
├── lancedb/                  # Vector DB
├── report/                   # Store healthcheck report(s) in PDF
└── logs/                     # Store logs
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | HuggingFace Transformers / llama-cpp-python | **GPU/CPU:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · **CPU:** [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M |
| Agent | LangGraph | ReAct loop: LLM selects tools → executes → observes → repeats or answers |
| Embeddings | SentenceTransformers | `nomic-ai/nomic-embed-text-v1.5` (local) |
| Vector DB | LanceDB (embedded) | Lightweight, works well with tabular (Arrow) + vector data together|
| Excel RAG | Column-aware ingestion | Sample xls: [ECS_KB.xls](/rag/sample/ECS_KB.xls) |
| K8s tools | kubernetes Python client | Read-only tools |
| API | FastAPI | REST + SSE streaming |
| Frontend | Single-file HTML/JS | Served by FastAPI at `/` |

---

## Kubernetes Tools

| Category | Tools |
|---|---|
| Pods | `get_pod_status`, `get_pod_logs`, `describe_pod`, `get_unhealthy_pods_detail`, `get_top_pods`, `get_pod_images`, `get_pod_tolerations`, `get_pod_containers_resources`, `get_pod_storage`, `get_pods_using_resource` |
| Namespaces | `get_namespace_status`, `get_namespace_resource_summary` |
| Nodes | `get_node_info`, `get_top_nodes`, `get_node_capacity`, `get_gpu_info`, `get_node_labels`, `get_node_taints`, `get_pods_on_node` |
| Workloads | `get_deployment`, `get_daemonset`, `get_statefulset`, `get_replicaset`, `get_adhoc_job_status`, `get_cronjob_status`, `get_hpa_status`, `get_pdb_status` |
| Storage | `get_pvc_status`, `describe_pvc`, `get_persistent_volumes`, `describe_pv`, `get_pv_usage`, `get_storage_classes`, `describe_sc`, `get_longhorn_settings`, `get_longhorn_node_status` |
| Networking | `get_service`, `get_ingress`, `get_endpoints`, `get_network_policy_status`, `get_ingress_traffic`, `get_coredns_health` |
| Cluster & Health | `run_cluster_health`, `get_cluster_version`, `get_control_plane_status`, `get_certificate_status`, `get_webhook_health`, `find_resource`, `get_crds` |
| Config | `get_configmap_list`, `get_secret_list`, `get_resource_quotas`, `get_limit_ranges` |
| RBAC | `get_serviceaccounts`, `get_cluster_role_bindings` |
| Events | `get_events` |
| Database | `exec_db_query` — read-only SQL inside DB pods (MySQL/PostgreSQL) |

---

## Example Queries

- Is the cluster doing OK?
- Where is grafana pod?
- List all pv that consume more than 80% of the allocated disk capacity
- Which node has a GPU available and in use?
- What storage classes are available in the cluster?
- List all replicaset in cdp-keda namespace
- Which nodes are ready and schedulable?
- Show all warning events in cdp namespace
- Decode cdp-private-installer-db-root-cert certificate in cdp ns
- Show the top 3 pods with the highest CPU and RAM usage over the past 3 months across the cluster
- Is user Dennis in cmlwb1 hogging resources for the past 20 days? (**Note:**  The tool is scripted to correlate the Cloudera AI username and the associated namespace)

---

## Quick Start

- **Python 3.12** is required.
- In an air-gapped environment, all Python libraries listed in `requirements.txt` must be pre-downloaded and hosted on an internal PyPI mirror or installed from local wheel files.

#### 1. Download the LLM and embedding models before starting:

##### Qwen3-8B
```bash
git clone https://huggingface.co/Qwen/Qwen3-8B ~/models/Qwen3-8B
```

##### Qwen3-8B-GGUF
```bash
git clone https://huggingface.co/Qwen/Qwen3-8B-GGUF ~/models/Qwen3-8B-GGUF
```

##### SentenceTransformers embedding model
```bash
git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 ~/models/nomic-embed-text-v1.5
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For NVIDIA GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Configure environment

The ECS Knowledge Bot app reads configuration values in the following order of precedence:

a. **Command-Line Arguments (CLI)**  
   Values passed when starting the app (e.g., `--port`, `--host`, `--model-dir`, `--embed-dir`) override all other settings.

b. **Settings File (`config/settings.json`)**  
   Persistent overrides applied in `config/settings.json` are loaded next. `MAX_NEW_TOKENS`, `LLM_TIMEOUT`, or `_KUBECTL_MAX_OUT` will automatically be updated according to the :gear: Settings in the Chatbot UI.

c. **Environment Variables (`env`)**  
   Variables like `KUBECONFIG_PATH`and `LOG_LEVEL` are loaded from environment if not overridden by CLI arguments.

d. **Defaults in (`config/config.py`)**  
   If a value is not provided via CLI or environment, the app falls back to the default specified in `config.py`.

> **Summary:**  
> **CLI > `config/settings.json` > `env` > Defaults in `config/config.py`**

Optional `env` configuration:

```ini
KUBECONFIG_PATH=~/kubeconfig
LOG_LEVEL=warning
```

##### Environment Variables References

| Setting | Controls | Default |
|---|---|---|
| `KUBECONFIG_PATH` | Path to kubeconfig (blank = in-cluster) | ~/kubeconfig |
| `LOG_LEVEL` | DEBUG shows full agentic loop trace | DEBUG |
| `GGUF_N_CTX` | Context window size (GGUF / CPU mode only) | 32768 |
| `GGUF_N_THREADS` | CPU threads for GGUF inference | all cores |

#### 4. Start the engine!

```bash
# Usage:
python3 app.py --host 0.0.0.0                               # bind address
python3 app.py --port 8080                                  # custom port
python3 app.py --model-dir  ~/models/Qwen3-8B               # LLM from local dir
python3 app.py --embed-dir  ~/models/nomic-embed-text-v1.5  # embeddings from local dir
```

**Note:** You may also run this chatbot via `Application` in Cloudera AI using the following Python `runme-cpu.py` script:

```python
import os
CDSW_APP_PORT=os.environ['CDSW_APP_PORT'] 
os.system("python ~/ECS-AI-Ops/app.py --host 127.0.0.1 --port $CDSW_APP_PORT --model-dir ~/models/Qwen3-8B --embed-dir ~/models/nomic-embed-text-v1.5 2>&1 > myapp.log")
```
<img src="https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/cml-ecsaiops.png" alt="cml1" width="700"/>

---

## Quick Start in Openshift

1. Create a `Dockerfile` on your machine (connected to Openshift cluster).
```dockerfile
FROM registry.redhat.io/ubi8/python-312
USER root
RUN dnf install -y git-lfs clang make cmake && \
    git lfs install && \
    dnf clean all
ENV CC=clang
ENV CXX=clang++
USER 1001
WORKDIR /opt/app-root/src
COPY . .
RUN pip install -r requirements.txt || true
RUN pip install llama-cpp-python
EXPOSE 8080
CMD ["python", "ocp_buildapp.py"]
```

2. Run this command to create a new build using your Dockerfile.
```bash
oc new-build https://github.com/dennislee22/ECS-AI-Ops --name=ecs-ai-app --strategy=docker --dockerfile="$(cat Dockerfile)"
```

3. After the build is completed, create the app pod.
```bash
oc new-app ecs-ai-app
```

4. Create a route YAML file `cpu-ecs-aiops-route.yaml`.
```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: ecs-ai-app
  annotations:
    haproxy.router.openshift.io/timeout: 5000s
spec:
  host: cpu-ecs-aiops.apps.yyy.com
  path: /
  to:
    kind: Service
    name: ecs-ai-app
    weight: 100
  port:
    targetPort: 8080-tcp
```

5. Apply the route.
```bash
oc apply -f cpu-ecs-aiops-route.yaml 
```

6. Open your browser and go to `http://cpu-ecs-aiops.apps.yyy.com`

---
## REST API

Interactive docs: **[/docs](http://localhost:8080/docs)** (Swagger) · **[/redoc](http://localhost:8080/redoc)**

| Method | Path | Description |
|---|---|---|
| POST | `/api/ask` | Ask the AI (blocking) |
| POST | `/api/tool` | Call a K8s tool directly |


#### Ask the ECS AI Ops Chatbot
```bash
curl -s -X POST http://localhost:8080/api/ask \
     -H "Content-Type: application/json" \
     -d '{"q":"list all pods with problems", "skip_synthesise": true}'
```

#### Call a specific ECS-K8s tool directly
```bash
curl -s -X POST http://localhost:8080/api/tool \
     -H "Content-Type: application/json" \
     -d '{"name":"get_pod_status","args":{"namespace":"cdp"}}'
```

---

## Hardware Sizing

| Inference mode | Model | Min. CPU cores| Min. RAM | Min. VRAM |
|---|---|---|---|---|
| GPU **(recommended)** | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8 | 32 GB | 25 GB |
| CPU | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 32 | 128 GB | — |
| CPU | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) Q4_K_M | 32 | 128 GB | — |

The application automatically detects available GPUs at startup and uses them if present. If no GPU is found, it falls back to CPU inference without any manual configuration.

- **GPU (recommended)** — use **[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** (bfloat16, HuggingFace Transformers). Responses typically complete in **5–30 secs**, depending on the query complexity. Requires an NVIDIA GPU with at least 25 GB VRAM (e.g. A100, RTX 3090/4090).

- **CPU** — use **[Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)**, specifically the **Q4_K_M** quantisation, via `llama-cpp-python`. Responses take **30 secs - x minutes**, depending on the query complexity.

You may examine the time taken to execute each query, using GPU, CPU, and CPU (GGUF):

| Inference | Skip Synthesis? | Log |
|---|---|---|
| GPU (A100 80GB PCIe) | No | [test-API-GPU-A100-80GB-output.log](https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/gpu_test_api.log) |
| CPU (GGUF)| Yes | [test-API-GGUF-output.log](https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/gguf_test_api_skip.log) |

---

## Security Notes

- All typed K8s tools are **read-only** by design.
- Secret values are hidden by default — toggle in ⚙ Settings → Security. Preference persists per browser.
- Restrict the env file: `chmod 600 env`

---

## Demo

| Inference using Nvidia GPU (Qwen3-8B) |
|---|
| <img src="https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/ecs-ai-ops-gpu.gif" width="700" /> |

| Inference using CPU (Qwen3-8B-GGUF) |
|---|
| <img src="https://raw.githubusercontent.com/dennislee22/huge-assets/main/ECS-AI-Ops-assets/ecs-ai-ops-gguf.gif" width="700" /> |

---

## Contact

[<img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" width="20" height="20"/>](https://www.linkedin.com/in/dennisleetw) dennisleetw
