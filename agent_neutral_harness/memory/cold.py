"""Cold tier: archive idle memories to an S3-compatible object store
(AWS S3, MinIO, Cloudflare R2, Backblaze B2, ...) and restore on demand.

"Idle" = not updated in ``days`` days. Archived rows are removed from the
hot SQLite/Chroma tiers and stored as one JSON object per row; the search
cascade transparently restores anything it finds here.

Requires the ``cold`` extra::

    pip install "agent-neutral-harness[cold]"

A custom ``client`` (anything implementing the handful of S3 methods used
here) can be injected -- the tests do this with an in-memory fake.
"""

import json
import logging
import time

from agent_neutral_harness.memory._similarity import cosine_similarity

log = logging.getLogger(__name__)


def _default_client(endpoint, access_key, secret_key, region):
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The cold tier needs: pip install 'agent-neutral-harness[cold]'") from exc
    kwargs = {"service_name": "s3"}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    if region:
        kwargs["region_name"] = region
    return boto3.client(**kwargs)


class ObjectStoreColdTier:
    def __init__(
        self,
        bucket: str,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        prefix: str = "memories/",
        accept_threshold: float = 0.5,
        client=None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.accept_threshold = accept_threshold
        self.client = client or _default_client(endpoint, access_key, secret_key, region)

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 - botocore raises many shapes
            self.client.create_bucket(Bucket=self.bucket)

    def _key(self, mem_id) -> str:
        return f"{self.prefix}{mem_id}.json"

    # ---------------------------------------------------------------- #
    def archive(self, vault, days: int = 90, dry_run: bool = False) -> str:
        cutoff = int(time.time()) - days * 86400
        rows = vault._rows_for_archive(cutoff)
        if not rows:
            return f"No memories idle for {days}+ days."
        if dry_run:
            preview = "\n".join(f"  #{r['id']} {r['scope']}/{r['type']}: {r['content'][:60]}"
                                for r in rows)
            return f"Would archive {len(rows)} memories:\n{preview}"

        self._ensure_bucket()
        for row in rows:
            key = self._key(row["id"])
            self.client.put_object(
                Bucket=self.bucket, Key=key,
                Body=json.dumps(row, default=str).encode("utf-8"),
                ContentType="application/json",
            )
            vault._mark_archived(row["id"], key)
        return f"Archived {len(rows)} memories to '{self.bucket}'."

    def restore(self, vault, mem_id: int) -> str:
        key = self._key(mem_id)
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001
            return f"Could not fetch memory #{mem_id} from cold storage: {exc}"
        vault._restore(json.loads(body))
        return f"Restored memory #{mem_id} from cold storage."

    def scan(self, query: str, embed_fn, threshold: float | None = None, limit: int = 3) -> list[dict]:
        threshold = self.accept_threshold if threshold is None else threshold
        qv = embed_fn(query)
        if qv is None:
            return []
        try:
            listing = self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
        except Exception:  # noqa: BLE001
            log.exception("cold tier listing failed")
            return []
        scored = []
        for obj in listing.get("Contents", [])[:100]:
            try:
                data = json.loads(
                    self.client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                )
            except Exception:  # noqa: BLE001
                continue
            sim = cosine_similarity(qv, embed_fn(data["content"]))
            if sim >= threshold:
                scored.append((sim, data))
        scored.sort(key=lambda t: -t[0])
        return [d for _, d in scored[:limit]]
