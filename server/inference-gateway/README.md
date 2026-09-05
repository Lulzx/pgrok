# Inference API

VPS-hosted OpenAI-compatible inference gateway for `inference.lulzx.space`.

## Public interface

- `GET /` — documentation and aggregate usage statistics
- `POST /api/token` — unauthenticated 30-day bearer-token issuance
- `GET /api/v1/models` — neutral `default` model descriptor
- `POST /api/v1/chat/completions` — chat, vision, and streaming
- `POST /api/v1/completions` — legacy text completions

The implementation stores only access-token hashes, issuance and request
timestamps, and aggregate request/token counters. Prompt and response content is
not persisted or written to application logs.

## Layout on the server

```text
/opt/inference-gateway/server.py
/opt/inference-gateway/static/
/etc/inference-gateway/provider.env
/var/lib/inference-gateway/state.sqlite3
/etc/systemd/system/inference-gateway.service
```

The service listens on `127.0.0.1:12341`; Caddy is its only public entrypoint.
The environment file is root-readable and must never be committed.

## Deployment

1. Install `server.py` and `static/` under `/opt/inference-gateway`.
2. Copy `provider.env.example` to `/etc/inference-gateway/provider.env`, fill
   the private values, and set mode `0600`.
3. Install `inference-gateway.service` in `/etc/systemd/system`.
4. Run `systemctl daemon-reload && systemctl enable --now inference-gateway`.
5. Add `Caddyfile.example` to the shared Caddy configuration and apply it with
   the VPS `caddy-apply` safety wrapper.

## Checks

```sh
curl http://127.0.0.1:12341/health
curl -X POST https://inference.lulzx.space/api/token
journalctl -u inference-gateway -f
```

The old laptop tunnel is not required and should remain disabled.
