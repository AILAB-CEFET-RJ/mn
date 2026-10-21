"""Gera um RASCUNHO de evaluation/queries.json com rótulos FRACOS de relevância.

Relevante = mesmo tipo de objeto (1º termo da descrição, ex.: "Maracá"). Isso é uma aproximação
para destravar a avaliação; REVISE MANUALMENTE antes de usar os números no TCC.
Preserva o arquivo existente se já houver revisão (use --force para sobrescrever).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "metadata.json"
OUT = Path(__file__).with_name("queries.json")

# Traduções para testar o viés de idioma do CLIP. REVISAR.
EN = {
    "Maracá": "maraca rattle", "Borduna": "wooden club", "Cocar": "feather headdress",
    "Apito": "whistle", "Cordão": "cord necklace", "Saia": "skirt", "Arco musical": "musical bow",
    "Peneira": "sieve", "Abanador": "fan", "Flecha": "arrow", "Arco": "bow", "Bodoque": "slingshot",
    "Crânio de macaco": "monkey skull", "Taquapu": "bamboo stamping tube", "Coroa": "crown",
}


def tipo(desc: str) -> str:
    return re.split(r"[/.]", desc or "")[0].strip()


def main():
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT.name} já existe; use --force para sobrescrever (perde revisões manuais).")
    meta = json.loads(META.read_text())
    items, imgs = {}, defaultdict(list)
    for r in meta:
        items.setdefault(r["item_id"], r)
        imgs[r["item_id"]].append(r)
    por_tipo = defaultdict(list)
    for iid, r in items.items():
        por_tipo[tipo(r["descricao"])].append(iid)

    texto = [{"id": f"t_{i:02d}", "texto_pt": t, "texto_en": EN.get(t), "relevantes": sorted(ids)}
             for i, (t, ids) in enumerate(sorted(por_tipo.items())) if t]
    imagem = []
    for t, ids in sorted(por_tipo.items()):
        if len(ids) < 2:
            continue  # sem outro item do mesmo tipo não há relevante
        for iid in ids:
            imagem.append({"id": f"i_{iid}", "item_id": iid, "image_id": imgs[iid][0]["image_id"],
                           "caminho_imagem": imgs[iid][0]["caminho_imagem"],
                           "relevantes": sorted(set(ids) - {iid})})
    OUT.write_text(json.dumps({
        "rotulos": "FRACOS: relevância = mesmo tipo de objeto; revisar manualmente",
        "consultas_texto": texto, "consultas_imagem": imagem,
    }, ensure_ascii=False, indent=1))
    print(f"{len(texto)} consultas de texto, {len(imagem)} de imagem")


if __name__ == "__main__":
    main()
