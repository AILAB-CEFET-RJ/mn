"""Gera embeddings CLIP (512-d, L2-normalizados) de forma incremental e retomável.

- Só processa imagens ainda não presentes no checkpoint (novos CSVs/pastas = só o novo).
- Decodificação JPEG em threads com `draft` (JPEGs grandes são o gargalo na CPU).
- Checkpoint a cada CKPT_EVERY lotes e em caso de erro/interrupção.
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
from load_catalog import PROCESSED, ROOT, load_catalog  # noqa: E402

log = logging.getLogger("clip_embed")
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH = int(os.getenv("BATCH_SIZE", "32"))
LIMIT = int(os.getenv("LIMIT", "0")) or None
CKPT_EVERY = int(os.getenv("CKPT_EVERY", "20"))
WORKERS = int(os.getenv("WORKERS", "4"))


def get_device() -> str:
    # ROCm expõe a GPU AMD via API "cuda". Fallback em CPU. Force com DEVICE=cpu.
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (448, 448))  # decodifica JPEG já reduzido (CLIP usa 224px)
        return im.convert("RGB")


def save_ckpt(path: Path, done: dict[str, np.ndarray]):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(list(done)), vecs=np.stack(list(done.values())))
    tmp.replace(path)  # escrita atômica: checkpoint nunca fica corrompido


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = load_catalog(LIMIT)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ckpt_path = PROCESSED / "image_embeddings.ckpt.npz"

    done: dict[str, np.ndarray] = {}
    if ckpt_path.exists():
        ck = np.load(ckpt_path, allow_pickle=False)
        done = dict(zip(ck["ids"].tolist(), ck["vecs"]))
        log.info("Checkpoint: %d embeddings já gerados", len(done))

    todo = [r for r in records if r["image_id"] not in done]
    failed: list[str] = []
    t0 = time.time()
    if todo:
        device = get_device()
        log.info("Dispositivo: %s | %d novas imagens", device, len(todo))
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        def load(r):
            try:
                return r, open_image(ROOT / r["caminho_imagem"])
            except Exception as e:  # imagem corrompida não derruba o lote
                log.error("Falha em %s: %s", r["image_id"], e)
                failed.append(r["image_id"])
                return r, None

        try:
            with ThreadPoolExecutor(WORKERS) as pool:
                for n, i in enumerate(tqdm(range(0, len(todo), BATCH), desc="CLIP")):
                    loaded = [x for x in pool.map(load, todo[i:i + BATCH]) if x[1] is not None]
                    if not loaded:
                        continue
                    with torch.no_grad():
                        inputs = processor(images=[im for _, im in loaded], return_tensors="pt").to(device)
                        vecs = model.get_image_features(**inputs)
                        if not isinstance(vecs, torch.Tensor):
                            vecs = vecs.pooler_output
                        vecs = torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy().astype("float32")
                    for (r, _), v in zip(loaded, vecs):
                        done[r["image_id"]] = v
                    if (n + 1) % CKPT_EVERY == 0:
                        save_ckpt(ckpt_path, done)
        finally:  # também salva em Ctrl+C / erro
            if done:
                save_ckpt(ckpt_path, done)
    elapsed = time.time() - t0

    final = [r for r in records if r["image_id"] in done]
    np.save(PROCESSED / "image_embeddings.npy", np.stack([done[r["image_id"]] for r in final]))
    (PROCESSED / "image_ids.json").write_text(json.dumps([r["image_id"] for r in final]))
    (PROCESSED / "metadata.json").write_text(json.dumps(final, ensure_ascii=False, indent=1))

    man_path = PROCESSED / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update({
        "image_model": MODEL_NAME, "dim": 512, "normalized": True,
        "count": len(final), "failed": failed, "versao_indexacao": "v1",
        "ultima_execucao_imagens": {
            "novas": len(todo) - len(failed), "segundos": round(elapsed, 1),
            "imagens_por_s": round((len(todo) - len(failed)) / elapsed, 2) if todo and elapsed else None,
            "dispositivo": get_device(), "batch": BATCH,
        },
    })
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    log.info("Concluído: %d embeddings (%d novos), %d falhas, %.1fs", len(final), len(todo) - len(failed),
             len(failed), elapsed)


if __name__ == "__main__":
    main()
