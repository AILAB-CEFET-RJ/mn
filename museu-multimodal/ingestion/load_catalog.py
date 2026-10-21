"""Importa o catálogo por imagem (Metadados_Imagens/MN_003.csv) e valida arquivos.

Formato real: CSV ';' utf-8-sig, uma linha por imagem; `codigo_acervo` agrupa
imagens do mesmo objeto (item_id); `nome_arquivo` aponta para a imagem.
"""
import csv
import json
import logging
import uuid
from pathlib import Path

log = logging.getLogger("load_catalog")

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Base Museu Nacional"
META_DIR = BASE / "Metadados_Imagens"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
NAMESPACE = uuid.UUID("6f1d6c3e-0a53-4c64-9d5e-2d6f0b9b7a11")


def image_id_from(name_no_ext: str) -> str:
    return name_no_ext.strip().replace(" ", "_")


def point_uuid(image_id: str) -> str:
    """UUID determinístico: reexecuções são idempotentes."""
    return str(uuid.uuid5(NAMESPACE, image_id))


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def load_catalog(limit: int | None = None) -> list[dict]:
    """Lê todos os CSVs de Metadados_Imagens/; imagens em Base/<nome_do_csv>/."""
    records, seen, skipped = [], set(), 0
    for csv_path in sorted(META_DIR.glob("*.csv")):
        img_dir = BASE / csv_path.stem
        rows = _read_rows(csv_path)
        for row in rows:
            if limit and len(records) >= limit:
                break
            nome = (row.get("nome_arquivo") or "").strip()
            item_id = (row.get("codigo_acervo") or "").strip()
            path = img_dir / nome
            if not nome or not item_id or not path.is_file():
                log.warning("Ignorado (sem arquivo/código): %r", nome)
                skipped += 1
                continue
            image_id = image_id_from(row.get("nome_sem_extensao") or path.stem)
            if image_id in seen:
                log.warning("Duplicado: %s", image_id)
                skipped += 1
                continue
            seen.add(image_id)
            tags = [t.strip() for t in (row.get("palavras_chave") or "").split("|") if t.strip()]
            records.append({
                "point_id": point_uuid(image_id),
                "item_id": item_id,
                "image_id": image_id,
                "titulo": row.get("titulo_colecao") or None,
                "descricao": row.get("descricao") or None,
                "tags": tags,
                "cidade": row.get("cidade") or None,
                "estado": row.get("estado") or None,
                "pais": row.get("pais") or None,
                "caminho_imagem": str(path.relative_to(ROOT)),
                "fonte_catalogo": f"Metadados_Imagens/{csv_path.name}",
                "versao_indexacao": "v1",
            })
            if limit and len(records) >= limit:
                break
    log.info("%d registros válidos, %d ignorados", len(records), skipped)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recs = load_catalog()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "metadata.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    print(f"metadata.json: {len(recs)} registros")
