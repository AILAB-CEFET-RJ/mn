"""Importa o catálogo por imagem (Metadados_Imagens/*.csv + Base*.csv) e valida arquivos.

Cruza os metadados técnicos da imagem (`Metadados_Imagens/MN_003.csv`) com os metadados
museológicos completos (`BaseMN_003.csv`), extraindo:
- identificação museológica (categoria), identificação local (termo indígena),
- descrição catalográfica física (`descricao_catalogo`) e legenda (`descricao`),
- material/suporte, técnica, função, dimensões, artesão, povo, aldeia,
- histórico administrativo, notas gerais e palavras-chave.
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


def _clean(val: str | None) -> str | None:
    if not val:
        return None
    s = " ".join(val.strip().split())
    return s or None


def _split_multi(val: str | None, sep: str = "||") -> list[str]:
    if not val:
        return []
    return [p.strip() for p in val.split(sep) if p.strip()]


def _load_base_catalog(stem: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Carrega Base<stem>.csv (ex.: BaseMN_003.csv) indexado por nome_arquivo e por codigo_acervo."""
    base_csv = BASE / f"Base{stem}.csv"
    by_file: dict[str, dict] = {}
    by_item: dict[str, dict] = {}
    if not base_csv.is_file():
        return by_file, by_item
    for r in _read_rows(base_csv):
        nf = (r.get("nome_arquivo") or "").strip()
        ca = (r.get("codigo_acervo") or "").strip()
        if nf:
            by_file[nf] = r
        if ca and ca not in by_item:
            by_item[ca] = r
    return by_file, by_item


def build_text_r2(m: dict) -> str:
    """Representação R2 (Seção 7.2): apenas a descrição textual catalográfica pura do objeto."""
    return _clean(m.get("descricao_catalogo")) or _clean(m.get("descricao")) or ""


def build_text_r4(m: dict) -> str:
    """Representação R4 (Seção 7.2): descrição + metadados catalográficos selecionados."""
    parts = [
        f"Categoria: {m['categoria']}" if m.get("categoria") else None,
        f"Denominação local/indígena: {m['identificacao_local']}" if m.get("identificacao_local") else None,
        f"Descrição física: {m['descricao_catalogo']}" if m.get("descricao_catalogo") else None,
        f"Descrição catalográfica: {m['descricao']}" if m.get("descricao") else None,
        f"Material e suporte: {m['material']}" if m.get("material") else None,
        f"Técnica: {m['tecnica']}" if m.get("tecnica") else None,
        f"Função e uso: {m['funcao']}" if m.get("funcao") else None,
        f"Povo: {m['povo']} ({m['auto_denominacao']})" if m.get("povo") and m.get("auto_denominacao")
        else (f"Povo: {m['povo']}" if m.get("povo") else None),
        f"Aldeia/Localidade: {m['aldeia']}" if m.get("aldeia") else None,
        f"Autor/Artesão: {m['artesao']}" if m.get("artesao") else None,
        f"Notas etnográficas e históricas: {m['notas_gerais']}" if m.get("notas_gerais") else None,
        f"Coleção: {m['titulo']}" if m.get("titulo") else None,
        f"Palavras-chave: {', '.join(m['tags'])}" if m.get("tags") else None,
    ]
    return ". ".join(p for p in parts if p)


def load_catalog(limit: int | None = None) -> list[dict]:
    """Lê todos os CSVs de Metadados_Imagens/, cruza com Base*.csv e valida imagens."""
    records, seen, skipped = [], set(), 0
    for csv_path in sorted(META_DIR.glob("*.csv")):
        stem = csv_path.stem
        img_dir = BASE / stem
        by_file, by_item = _load_base_catalog(stem)
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

            cat = by_file.get(nome) or by_item.get(item_id) or {}
            tags = [t.strip() for t in (row.get("palavras_chave") or "").split("|") if t.strip()]
            materiais = _split_multi(cat.get("catalogo_material_suporte"))
            funcoes = _split_multi(cat.get("catalogo_função"))

            aldeia = _clean(cat.get("catalogo_aldeia"))
            estado = _clean(row.get("estado"))
            if not estado and aldeia and "-MS" in aldeia.replace(" ", ""):
                estado = "MS"
            pais = _clean(row.get("pais")) or ("Brasil" if "Brasil" in (row.get("descricao") or "") else None)

            rec = {
                "point_id": point_uuid(image_id),
                "item_id": item_id,
                "image_id": image_id,
                "categoria": _clean(cat.get("catalogo_identificação_museológica")),
                "identificacao_local": _clean(cat.get("catalogo_identificação_local")),
                "titulo": _clean(cat.get("catalogo_nome_da_coleção")) or _clean(row.get("titulo_colecao")),
                "descricao_catalogo": _clean(cat.get("catalogo_descrição")),
                "descricao": _clean(row.get("descricao")),
                "material": _clean(cat.get("catalogo_material_suporte")),
                "materiais_lista": materiais,
                "tecnica": _clean(cat.get("catalogo_técnica")),
                "funcao": _clean(cat.get("catalogo_função")),
                "funcoes_lista": funcoes,
                "dimensoes": _clean(cat.get("catalogo_dimensões")),
                "artesao": _clean(cat.get("catalogo_autor_artesão")),
                "povo": _clean(cat.get("catalogo_povo")),
                "auto_denominacao": _clean(cat.get("catalogo_auto_denominação")),
                "aldeia": aldeia,
                "data_coleta": _clean(cat.get("catalogo_data_de_coleta")),
                "data_fabricacao": _clean(cat.get("catalogo_data_de_fabricação")),
                "historico": _clean(cat.get("catalogo_histórico_administrativo")),
                "notas_gerais": _clean(cat.get("catalogo_notas_gerais")),
                "tags": tags,
                "cidade": _clean(row.get("cidade")),
                "estado": estado,
                "pais": pais,
                "largura_pixels": int(row["largura_pixels"]) if (row.get("largura_pixels") or "").isdigit() else None,
                "altura_pixels": int(row["altura_pixels"]) if (row.get("altura_pixels") or "").isdigit() else None,
                "megapixels": float(row["megapixels"]) if row.get("megapixels") else None,
                "tamanho_mb": float(row["tamanho_mb"]) if row.get("tamanho_mb") else None,
                "orientacao": _clean(row.get("orientacao_imagem")),
                "caminho_imagem": str(path.relative_to(ROOT)),
                "fonte_catalogo": f"Base{stem}.csv + Metadados_Imagens/{csv_path.name}",
                "versao_indexacao": "v2",
            }
            rec["texto_r2"] = build_text_r2(rec)
            rec["texto_r4"] = build_text_r4(rec)
            records.append(rec)
            if limit and len(records) >= limit:
                break
    log.info("%d registros válidos, %d ignorados", len(records), skipped)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recs = load_catalog()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "metadata.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    print(f"metadata.json: {len(recs)} registros ({len({r['item_id'] for r in recs})} objetos únicos)")
