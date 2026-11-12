"""Constrói `evaluation/queries.json` cobrindo as Tarefas de Recuperação da Seção 7.3 do PDF.

Tarefas definidas:
1. `categoria`: Recuperar objetos pertencentes à mesma categoria (`catalogo_identificação_museológica`).
2. `linguagem_natural`: Consultas livres e semânticas por função, uso ritual, material ou técnica
   (incluindo os exemplos literais da proposta do professor, Seções 2 e 6.1).
3. `imagem_imagem`: Recuperar objetos semelhantes a partir de uma imagem de consulta (leave-one-out).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "metadata.json"
OUT = Path(__file__).with_name("queries.json")

EN_CAT = {
    "Maracá": "maraca gourd rattle musical instrument",
    "Borduna": "carved wooden war club weapon",
    "Cocar": "indigenous feather and woven headdress crown",
    "Apito": "wooden whistle flute musical instrument",
    "Cordão": "seed and wool necklace cord",
    "Saia": "woven cotton and feather skirt",
    "Arco": "wooden bow weapon",
    "Arco de boca": "wooden musical bow instrument",
    "Peneira": "woven plant fiber sieve basket",
    "Abanador": "woven palm leaf fan",
    "Flecha": "wooden arrow weapon",
    "Crânio": "animal monkey skull bone",
    "Taquara": "bamboo stamping tube rhythm instrument",
}


def build_semantic_queries(items: dict[str, dict]) -> list[dict]:
    """Consultas livres em linguagem natural (Seções 2, 6.1 e 7.3 do PDF)."""
    by_cat = defaultdict(list)
    for iid, r in items.items():
        by_cat[r.get("categoria")].append(iid)

    def ids_for_cats(*cats):
        out = []
        for c in cats:
            out.extend(by_cat.get(c, []))
        return sorted(set(out))

    def ids_where(pred):
        return sorted(iid for iid, r in items.items() if pred(r))

    raw = [
        {
            "id": "ln_01",
            "tarefa": "linguagem_natural",
            "texto_pt": "instrumentos musicais utilizados em cerimônias",
            "texto_en": "musical instruments used in ceremonies and rituals",
            "criterio": "Objetos musicais e sonoros rituais (Maracá, Arco de boca, Taquara, Apito)",
            "relevantes": ids_for_cats("Maracá", "Arco de boca", "Taquara", "Apito"),
        },
        {
            "id": "ln_02",
            "tarefa": "linguagem_natural",
            "texto_pt": "objetos utilizados em cerimônias religiosas",
            "texto_en": "objects used in religious ceremonies and rituals",
            "criterio": "Objetos com função Religioso / Ritualístico no catálogo",
            "relevantes": ids_where(
                lambda r: any(x in (r.get("funcao") or "") for x in ("Religioso", "Ritual"))
            ),
        },
        {
            "id": "ln_03",
            "tarefa": "linguagem_natural",
            "texto_pt": "armas de madeira talhada usadas na retomada de território e defesa",
            "texto_en": "carved wooden weapons used in territory reclamation and defense",
            "criterio": "Objetos com função Arma || Retomada (Bordunas, Flecha, Arco)",
            "relevantes": ids_where(lambda r: "Arma" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_04",
            "tarefa": "linguagem_natural",
            "texto_pt": "instrumentos de sopro usados por xamãs para reza e comunicação",
            "texto_en": "wooden wind whistle instruments used by shamans for prayer and communication",
            "criterio": "Objetos com função Comunicação (Apitos / Mimby)",
            "relevantes": ids_where(lambda r: "Comunicação" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_05",
            "tarefa": "linguagem_natural",
            "texto_pt": "adornos de sementes e lã usados cruzados no peito para proteção",
            "texto_en": "seed and wool necklaces worn crossed on the chest for protection",
            "criterio": "Objetos com função Proteção (Cordões / Jeha-saha)",
            "relevantes": ids_where(lambda r: "Proteção" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_06",
            "tarefa": "linguagem_natural",
            "texto_pt": "utensílios domésticos trançados em fibras vegetais para alimentos ou abanar o fogo",
            "texto_en": "household utensils woven from plant fibers for food preparation or fanning fire",
            "criterio": "Objetos com função Utilitário (Peneira e Abanador)",
            "relevantes": ids_where(lambda r: "Utilitário" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_07",
            "tarefa": "linguagem_natural",
            "texto_pt": "vestimentas e adornos de cabeça usados em encontros políticos e rituais",
            "texto_en": "garments skirts and headdresses worn in political meetings and rituals",
            "criterio": "Objetos com função Político (Saias e Cocares)",
            "relevantes": ids_where(lambda r: "Político" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_08",
            "tarefa": "linguagem_natural",
            "texto_pt": "chocalhos feitos de cabaça com fios de lã coloridos",
            "texto_en": "gourd rattles with colorful wool threads",
            "criterio": "Objetos cujo material principal é Cabaça (Maracás / Mbaraka)",
            "relevantes": ids_where(lambda r: "Cabaça" in (r.get("material") or "")),
        },
        {
            "id": "ln_09",
            "tarefa": "linguagem_natural",
            "texto_pt": "objeto osteológico usado para fins curativos e terapêuticos",
            "texto_en": "bone skull object used for healing and therapeutic purposes",
            "criterio": "Objetos com função Curativo || Terapeutico (Crânio / Ka'i-ankãgue)",
            "relevantes": ids_where(lambda r: "Curativo" in (r.get("funcao") or "")),
        },
        {
            "id": "ln_10",
            "tarefa": "linguagem_natural",
            "texto_pt": "instrumento de corda única com dois arcos de madeira que se cruzam",
            "texto_en": "single string musical instrument with two crossed wooden bows",
            "criterio": "Arcos musicais / Arco de boca (Yvyra-pa'i)",
            "relevantes": ids_for_cats("Arco de boca"),
        },
    ]
    return raw


def main():
    meta = json.loads(META.read_text())
    items, imgs = {}, defaultdict(list)
    for r in meta:
        items.setdefault(r["item_id"], r)
        imgs[r["item_id"]].append(r)

    por_cat = defaultdict(list)
    for iid, r in items.items():
        cat = r.get("categoria") or "Sem categoria"
        por_cat[cat].append(iid)

    consultas_categoria = [
        {
            "id": f"cat_{i:02d}",
            "tarefa": "categoria",
            "texto_pt": cat,
            "texto_en": EN_CAT.get(cat, cat),
            "criterio": f"catalogo_identificação_museológica == '{cat}'",
            "relevantes": sorted(ids),
        }
        for i, (cat, ids) in enumerate(sorted(por_cat.items()), start=1)
    ]

    consultas_ln = build_semantic_queries(items)

    consultas_imagem = []
    for cat, ids in sorted(por_cat.items()):
        if len(ids) < 2:
            continue
        for iid in sorted(ids):
            consultas_imagem.append({
                "id": f"img_{iid}",
                "tarefa": "imagem_imagem",
                "categoria": cat,
                "item_id": iid,
                "image_id": imgs[iid][0]["image_id"],
                "caminho_imagem": imgs[iid][0]["caminho_imagem"],
                "relevantes": sorted(set(ids) - {iid}),
            })

    payload = {
        "protocolo": (
            "Seção 7.3 do PDF: Tarefa 1 = busca por categoria oficial (catalogo_identificação_museológica); "
            "Tarefa 2 = consultas livres e semânticas em linguagem natural (função, material, uso ritual); "
            "Tarefa 3 = busca imagem→imagem leave-one-out."
        ),
        "consultas_categoria": consultas_categoria,
        "consultas_linguagem_natural": consultas_ln,
        "consultas_texto": consultas_categoria + consultas_ln,
        "consultas_imagem": consultas_imagem,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"queries.json gerado: {len(consultas_categoria)} categoria + "
        f"{len(consultas_ln)} linguagem natural + {len(consultas_imagem)} imagem→imagem"
    )


if __name__ == "__main__":
    main()
