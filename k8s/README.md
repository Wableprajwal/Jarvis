# Kubernetes manifests — design artifacts

**These have not been applied to a live cluster.** They describe how Jarvis would be deployed and exist to make the
deployment argument concrete and reviewable. They are structurally validated (YAML parses, required fields present, each
Service's selector matches a Deployment's pod labels) but not schema-validated against a live API server.

## What's here

| File | Contents |
|---|---|
| `configmap.yaml` | non-secret configuration — model per role, budgets, paths |
| `deployment.yaml` | backend (2 replicas) and frontend Deployments |
| `service.yaml` | ClusterIP Services for both |

## Assumed to exist in the cluster (deliberately not modelled here)

- **Postgres with pgvector** — a managed instance in a real deployment, reached via `DATABASE_URL` from the Secret. It is
  stateful and backed up on a different lifecycle than this app, so it does not belong in these manifests.
- **A model server** — the ConfigMap points `OLLAMA_HOST` at `ollama.models.svc.cluster.local`. Inference wants GPU
  nodes, its own scaling behaviour and a multi-GB model cache; it is a separate workload, which is the point of
  `ModelProvider` being an interface.
- **An Ingress + TLS + authentication** in front of `jarvis-frontend`. This build has no authN of its own
  (`user_id` is a namespacing key, not identity — see DECISIONS.md), so exposing it publicly without one would be wrong.

## The Secret

Never committed. Create it out of band:

```bash
kubectl create secret generic jarvis-secrets \
  --from-literal=database-url='postgresql://jarvis:...@postgres:5432/jarvis' \
  --from-literal=gmail-token='...' \
  --from-literal=calendar-token='...'
```

The deployment marks the tool tokens `optional: true`, so the app starts without them and the calendar/email tools run
against their fakes. Secrets reach the process environment and are read only by `app/tools/credentials.py`, which hands
them to tools as closures — they never enter a prompt, a tool schema or a trace payload.

## What changes from the local build

- **`code_exec` must not run in the backend's own pod.** Locally it uses a subprocess sandbox (Seatbelt/bwrap/unshare).
  In a cluster it should become a short-lived Job or a container-per-execution with `--network none`, a read-only
  rootfs and CPU/memory/time limits — or gVisor/Firecracker — so untrusted generated code never shares a kernel trust
  boundary with the harness that holds credentials. The Deployment carries a comment where that boundary belongs.
- **Pending confirmations are in-process memory** today; they would move to the database so an approval survives a
  restart and works across replicas.
- **Sessions are sticky-free but memory is not cached**, so 2 replicas are safe; adding a cache would need shared state.
- Model-loading makes startup slow (embedding + reranker models), hence `readinessProbe.initialDelaySeconds: 20`. Baking
  the model cache into the image or a PVC would cut that.
