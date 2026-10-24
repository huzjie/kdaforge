# -*- coding: utf-8 -*-
"""Zero-dependency HTTP server exposing the model over /v1/*.

Uses only the Python standard library (http.server). Endpoints:

  * GET  /health            -> {"status": "ok"}
  * GET  /v1/models         -> declared model card
  * POST /v1/infer          -> one forward pass (tiny tensors)
  * GET  /v1/memory         -> KV memory profile of the hybrid stack
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..config import load_config
from ..model import KDAForgeModel


def _build_model():
    cfg = load_config()
    return cfg, KDAForgeModel(cfg)


class Handler(BaseHTTPRequestHandler):
    model_state = {}

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            cfg = Handler.model_state.get("cfg")
            if cfg is None:
                cfg, _ = _build_model()
                Handler.model_state["cfg"] = cfg
            d = cfg.declared
            return self._json({
                "id": "kdaforge", "total_params_b": d.total_params_b,
                "active_params_b": d.active_params_b, "n_experts": d.n_experts,
                "top_k": d.top_k, "context_len": d.context_len,
                "kda_layers": d.kda_layers, "mla_layers": d.mla_layers,
            })
        if self.path == "/v1/memory":
            cfg, model = _build_model()
            seq_len = 1000000
            full, actual = model.attn.kv_memory(seq_len)
            return self._json({"seq_len": seq_len, "full_kv_floats": full,
                               "hybrid_kv_floats": actual,
                               "ratio": round(full / actual, 2)})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/infer":
            cfg, model = _build_model()
            n = self._read_len()
            body = self.rfile.read(n) if n else b""
            try:
                req = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                req = {}
            token_ids = req.get("token_ids", [0, 1, 2, 3])
            x = model.embed(token_ids)
            out = model.forward(x)
            return self._json({"token_ids": token_ids,
                               "output_dim": len(out[0]) if out else 0,
                               "n_layers": model.attn.n_layers})
        return self._json({"error": "not found"}, 404)

    def _read_len(self):
        try:
            return int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return 0

    def log_message(self, *args):
        pass


def run(host="127.0.0.1", port=8321):
    srv = HTTPServer((host, port), Handler)
    print("kdaforge serving on http://%s:%d" % (host, port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    run()
