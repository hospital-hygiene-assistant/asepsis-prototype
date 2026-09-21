"""
Find the Bedrock model identifier to put in config.json, and check what the
account can actually call.

The exact identifier for a given model is an account- and region-specific
fact: foundation models, cross-region inference profiles and marketplace
deployments all have different id shapes, and a wrong id fails at call time
with an unhelpful error. This lists what the credential can see so the id is
read off the account rather than guessed.

    export AWS_BEARER_TOKEN_BEDROCK=...          # or normal AWS credentials
    python3 llm-baseline/discover_models.py --filter gpt

Prints matching foundation models and inference profiles, with the field to
copy into config.json as `model_id`. Add --all to list everything, and
--region to look somewhere other than the configured region.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="", help="case-insensitive substring to match")
    ap.add_argument("--region", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print("boto3 is not installed:\n"
              "    python3 -m pip install -r requirements-baseline.txt", file=sys.stderr)
        return 1

    cfg = load_config()
    region = args.region or cfg.get("region") or "us-east-1"
    if not (os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")):
        print("no Bedrock credential in the environment. Set AWS_BEARER_TOKEN_BEDROCK "
              "(the Bedrock API key) or normal AWS credentials first.", file=sys.stderr)
        return 1

    needle = args.filter.lower()
    client = boto3.client("bedrock", region_name=region)

    def show(label: str, rows: list[tuple[str, str]]) -> None:
        print(f"\n=== {label} ({region}) — {len(rows)} match(es)")
        for ident, detail in rows:
            print(f"  {ident}")
            if detail:
                print(f"      {detail}")

    try:
        models = client.list_foundation_models().get("modelSummaries", [])
    except (BotoCoreError, ClientError) as exc:
        print(f"list_foundation_models failed: {exc}", file=sys.stderr)
        models = []
    rows = []
    for m in models:
        ident = m.get("modelId", "")
        blob = f"{ident} {m.get('modelName', '')} {m.get('providerName', '')}".lower()
        if args.all or (needle and needle in blob):
            streaming = "streaming" if m.get("responseStreamingSupported") else "no-streaming"
            rows.append((ident, f"{m.get('providerName', '?')} / {m.get('modelName', '?')} "
                                f"[{','.join(m.get('inferenceTypesSupported') or [])}] {streaming}"))
    show("foundation models", rows)

    try:
        profiles = client.list_inference_profiles().get("inferenceProfileSummaries", [])
    except (BotoCoreError, ClientError) as exc:
        print(f"\nlist_inference_profiles failed: {exc}", file=sys.stderr)
        profiles = []
    rows = []
    for p in profiles:
        ident = p.get("inferenceProfileId", "")
        blob = f"{ident} {p.get('inferenceProfileName', '')}".lower()
        if args.all or (needle and needle in blob):
            rows.append((ident, f"{p.get('inferenceProfileName', '?')} "
                                f"({p.get('type', '?')}, {p.get('status', '?')})"))
    show("inference profiles", rows)

    try:
        markets = client.list_marketplace_model_endpoints().get(
            "marketplaceModelEndpoints", [])
    except Exception:
        markets = []
    rows = []
    for e in markets:
        ident = e.get("endpointArn", "")
        blob = f"{ident} {e.get('modelSourceIdentifier', '')}".lower()
        if args.all or (needle and needle in blob):
            rows.append((ident, e.get("modelSourceIdentifier", "")))
    if rows:
        show("marketplace endpoints", rows)

    print("\nCopy the identifier you want into llm-baseline/config.json as \"model_id\".")
    print("A cross-region inference profile id (us.<provider>.<model>) is usually the "
          "one that works when the bare foundation-model id is refused.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
