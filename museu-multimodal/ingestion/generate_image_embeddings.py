"""Gera embeddings visuais CLIP (512-d) e SigLIP Multilíngue (768-d) de forma incremental.

Atende à Seção 6.4 (Comparação de modelos multimodais: CLIP vs SigLIP) e Etapa 2 (R1).
- Só processa imagens ainda não presentes no checkpoint.
- Decodificação JPEG em threads com `draft`.
- Escrita atômica de checkpoints. Defina SKIP_SIGLIP=1 para rodar apenas o CLIP.
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from app.config import CLIP_NAME as CLIP_MODEL, SIGLIP_NAME as SIGLIP_MODEL  # noqa: E402
from ingestion.load_catalog import PROCESSED, ROOT, load_catalog  # noqa: E402

log = logging.getLogger("image_embed")
BATCH = int(os.getenv("BATCH_SIZE", "32"))
LIMIT = int(os.getenv("LIMIT", "0")) or None
CKPT_EVERY = int(os.getenv("CKPT_EVERY", "20"))
WORKERS = int(os.getenv("WORKERS", "4"))


def get_device() -> str:
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        return im.convert("RGB")


def save_ckpt(path: Path, done: dict[str, np.ndarray]):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(list(done)), vecs=np.stack(list(done.values())))
    tmp.replace(path)


def encode_dataset(records: list[dict], model_name: str, ckpt_path: Path, out_path: Path,
                   is_siglip: bool = False) -> tuple[dict[str, np.ndarray], list[str], dict]:
    done: dict[str, np.ndarray] = {}
    if ckpt_path.exists():
        ck = np.load(ckpt_path, allow_pickle=False)
        done = dict(zip(ck["ids"].tolist(), ck["vecs"]))
        log.info("[%s] Checkpoint: %d embeddings já gerados", model_name, len(done))

    todo = [r for r in records if r["image_id"] not in done]
    failed: list[str] = []
    device = get_device()
    t0 = time.time()

    if todo:
        log.info("[%s] Dispositivo: %s | %d novas imagens", model_name, device, len(todo))
        raw_model: Any = (
            AutoModel.from_pretrained(model_name) if is_siglip else CLIPModel.from_pretrained(model_name)
        )
        model: Any = raw_model.to(device).eval()
        processor: Any = (
            AutoProcessor.from_pretrained(model_name) if is_siglip else CLIPProcessor.from_pretrained(model_name)
        )

        def load(r: dict) -> tuple[dict, Image.Image | None]:
            try:
                return r, open_image(ROOT / r["caminho_imagem"])
            except Exception as e:
                log.error("Falha em %s: %s", r["image_id"], e)
                failed.append(r["image_id"])
                return r, None

        try:
            with ThreadPoolExecutor(WORKERS) as pool:
                for n, i in enumerate(tqdm(range(0, len(todo), BATCH), desc=model_name.split("/")[-1])):
                    loaded: list[tuple[dict, Image.Image]] = [
                        (r, im) for r, im in pool.map(load, todo[i:i + BATCH]) if im is not None
                    ]
                    if not loaded:
                        continue
                    with torch.no_grad():
                        inputs = processor(images=[im for _, im in loaded], return_tensors="pt").to(device)
                        raw_out = model.get_image_features(**inputs)
                        tensor = raw_out if isinstance(raw_out, torch.Tensor) else raw_out.pooler_output
                        assert isinstance(tensor, torch.Tensor)
                        vecs = torch.nn.functional.normalize(tensor, dim=-1).cpu().numpy().astype("float32")
                    for (r, _), v in zip(loaded, vecs):
                        done[r["image_id"]] = v
                    if (n + 1) % CKPT_EVERY == 0:
                        save_ckpt(ckpt_path, done)
        finally:
            if done:
                save_ckpt(ckpt_path, done)

    elapsed = time.time() - t0
    final = [r for r in records if r["image_id"] in done]
    arr = np.stack([done[r["image_id"]] for r in final])
    np.save(out_path, arr)
    stats = {
        "modelo": model_name,
        "dim": int(arr.shape[1]),
        "novas": len(todo) - len(failed),
        "segundos": round(elapsed, 1),
        "imagens_por_s": round((len(todo) - len(failed)) / elapsed, 2) if todo and elapsed else None,
        "dispositivo": device,
        "batch": BATCH,
    }
    return done, failed, stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = load_catalog(LIMIT)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    done_clip, failed, clip_stats = encode_dataset(
        records,
        CLIP_MODEL,
        PROCESSED / "image_embeddings.ckpt.npz",
        PROCESSED / "image_embeddings.npy",
        is_siglip=False,
    )
    final = [r for r in records if r["image_id"] in done_clip]

    siglip_stats = None
    if os.getenv("SKIP_SIGLIP") != "1":
        _, _, siglip_stats = encode_dataset(
            final,
            SIGLIP_MODEL,
            PROCESSED / "siglip_image_embeddings.ckpt.npz",
            PROCESSED / "siglip_image_embeddings.npy",
            is_siglip=True,
        )

    (PROCESSED / "image_ids.json").write_text(json.dumps([r["image_id"] for r in final]), encoding="utf-8")
    (PROCESSED / "metadata.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")

    man_path = PROCESSED / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    man.update({
        "image_model": CLIP_MODEL,
        "dim": 512,
        "siglip_model": SIGLIP_MODEL if siglip_stats else None,
        "siglip_dim": siglip_stats["dim"] if siglip_stats else None,
        "normalized": True,
        "count": len(final),
        "failed": failed,
        "versao_indexacao": "v2",
        "ultima_execucao_imagens": clip_stats,
        "ultima_execucao_siglip": siglip_stats,
    })
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("Concluído: %d imagens processadas", len(final))


if __name__ == "__main__":
    main()
