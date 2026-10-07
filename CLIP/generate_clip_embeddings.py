import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Configurações de diretórios
CLIP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLIP_DIR.parent
IMG_DIR = PROJECT_ROOT / "Base Museu Nacional" / "MN_003"
CSV_PATH = PROJECT_ROOT / "Base Museu Nacional" / "Metadados_Imagens" / "MN_003.csv"
OUTPUT_DIR = CLIP_DIR / "Embeddings_CLIP_MN_003"

# Modelo CLIP (OpenAI CLIP ViT-B/32)
MODEL_NAME = "openai/clip-vit-base-patch32"

def extract_tensor(features):
    """Extrai o tensor de embedding de forma compatível com diferentes versões do Transformers."""
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if isinstance(features, torch.Tensor):
        return features
    raise ValueError(f"Não foi possível extrair tensor de: {type(features)}")

def main():
    print(f"Diretório de Imagens: {IMG_DIR}")
    print(f"Arquivo CSV: {CSV_PATH}")
    print(f"Diretório de Saída: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Detecção de dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando modelo CLIP '{MODEL_NAME}' no dispositivo: {device}...")
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # Leitura do CSV
    with open(CSV_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    print(f"Total de registros no CSV: {len(rows)}")

    # Agrupar registros por codigo_acervo
    objects_data = defaultdict(lambda: {
        "codigo_acervo": "",
        "descricao": "",
        "titulo_colecao": "",
        "imagens": []
    })

    for row in rows:
        code = row.get("codigo_acervo", "").strip()
        if not code:
            continue
        
        obj = objects_data[code]
        obj["codigo_acervo"] = code
        obj["descricao"] = row.get("descricao", "").strip()
        obj["titulo_colecao"] = row.get("titulo_colecao", "").strip()
        obj["imagens"].append({
            "nome_arquivo": row.get("nome_arquivo", "").strip(),
            "largura": row.get("largura_pixels", "").strip(),
            "altura": row.get("altura_pixels", "").strip(),
            "palavras_chave": row.get("palavras_chave", "").strip(),
        })

    print(f"Total de objetos únicos identificados: {len(objects_data)}")

    resumo_geral = []

    for code, obj_info in tqdm(objects_data.items(), desc="Processando objetos"):
        obj_dir = OUTPUT_DIR / code
        obj_dir.mkdir(parents=True, exist_ok=True)

        desc_text = obj_info["descricao"]
        
        # 1. Gerar e salvar embedding do texto da descrição
        text_emb = None
        if desc_text:
            text_inputs = processor(
                text=[desc_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            with torch.no_grad():
                raw_text_features = model.get_text_features(**text_inputs)
                text_features = extract_tensor(raw_text_features)
                # Normalizar vetor L2 (padrão CLIP para busca por cosseno)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                text_emb = text_features.cpu().numpy().squeeze(0)
            
            np.save(obj_dir / "descricao_embedding.npy", text_emb)

        # 2. Gerar e salvar embeddings de cada imagem do objeto
        imagens_processadas = []
        for img_info in obj_info["imagens"]:
            img_name = img_info["nome_arquivo"]
            img_path = IMG_DIR / img_name
            
            if not img_path.exists():
                print(f"[Aviso] Imagem não encontrada: {img_path}")
                continue

            try:
                with Image.open(img_path) as image:
                    image = image.convert("RGB")
                    image_inputs = processor(images=image, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        raw_image_features = model.get_image_features(**image_inputs)
                        image_features = extract_tensor(raw_image_features)
                        # Normalizar vetor L2
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                        img_emb = image_features.cpu().numpy().squeeze(0)

                # Salva embedding da imagem: ex: MN 003-33.npy
                stem_name = Path(img_name).stem
                npy_filename = f"{stem_name}.npy"
                np.save(obj_dir / npy_filename, img_emb)
                
                imagens_processadas.append({
                    "arquivo_imagem": img_name,
                    "arquivo_embedding": npy_filename,
                    "embedding_shape": list(img_emb.shape),
                    "dimensao": int(img_emb.shape[0])
                })
            except Exception as e:
                print(f"[Erro] Falha ao processar {img_path}: {e}")

        # 3. Salvar metadata.json individual do objeto
        metadata = {
            "codigo_acervo": code,
            "descricao": desc_text,
            "titulo_colecao": obj_info["titulo_colecao"],
            "modelo_clip": MODEL_NAME,
            "dimensao_vetores": int(text_emb.shape[0]) if text_emb is not None else (int(imagens_processadas[0]["dimensao"]) if imagens_processadas else 512),
            "total_imagens": len(imagens_processadas),
            "arquivo_embedding_descricao": "descricao_embedding.npy" if text_emb is not None else None,
            "imagens": imagens_processadas
        }

        with open(obj_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        resumo_geral.append({
            "codigo_acervo": code,
            "pasta": str(obj_dir.relative_to(PROJECT_ROOT)),
            "total_imagens": len(imagens_processadas),
            "descricao": desc_text[:120] + ("..." if len(desc_text) > 120 else "")
        })

    # Salva índice geral de todos os objetos processados
    index_path = OUTPUT_DIR / "indice_objetos.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "modelo_clip": MODEL_NAME,
            "total_objetos": len(resumo_geral),
            "objetos": resumo_geral
        }, f, ensure_ascii=False, indent=2)

    print(f"\nConcluído com sucesso! {len(resumo_geral)} objetos processados.")
    print(f"Embeddings salvos em: {OUTPUT_DIR}")
    print(f"Índice geral salvo em: {index_path}")

if __name__ == "__main__":
    main()
