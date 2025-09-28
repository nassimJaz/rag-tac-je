from typing import List, Optional
import re
import json
from haystack.dataclasses.document import Document

def extract_source_numbers(query: str, documents: List[Document]) -> Optional[List[int]]:
    """
    Extrait les numéros de sources utilisés dans une question en analysant les références [n]
    et en les mappant aux documents disponibles.
    
    Args:
        query: La question contenant des références de sources [1], [2], etc.
        documents: Liste des documents disponibles pour vérification
        
    Returns:
        Liste des numéros de sources utilisés, ou None si aucune source trouvée
    """
    try:
        # 1. Extraire tous les numéros de sources de la question
        source_matches = re.findall(r'\[(\d+)\]', query)
        source_numbers = [int(num) for num in source_matches]
        
        if not source_numbers:
            return None
        
        # 2. Filtrer les numéros valides (dans la plage des documents)
        valid_sources = []
        for num in source_numbers:
            if 1 <= num <= len(documents):
                valid_sources.append(num)
            else:
                print(f"⚠️  Source [{num}] ignorée - hors plage (1-{len(documents)})")
        
        # 3. Dédupliquer et trier
        unique_sources = list(set(valid_sources))
        unique_sources.sort()
        
        return unique_sources if unique_sources else None
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des sources: {e}")
        return None

def format_sources(docs: List[Document]) -> List[str]:
    """
    Formate les documents en liste de sources lisibles
    """
    all_sources = [] 
    for i, doc in enumerate(docs, start=1):
        ref = f"[{i}]"
        src = doc.meta.get("source", "inconnu")
        page = doc.meta.get("page")
        label = f"{src}" + (f", page {page}" if page is not None else "")
        all_sources.append(f"{ref} {label}")
    return all_sources

def extracts_sources(query:str, documents: List[Document]):
    """
    Extrait les sources formatées basées sur les numéros de sources
    """
    all_sources = format_sources(documents)
    used_sources = []
    source_numbers = extract_source_numbers(query=query, documents=documents)
    if(source_numbers is None): return None, None
    for n in source_numbers:
        try:
            idx = int(n) - 1
            if 0 <= idx < len(all_sources):
                used_sources.append(all_sources[idx])
        except ValueError:
            continue
    
    return used_sources

def format_json_to_human_text_simple(parsed_json: dict) -> str:
    """
    Version simple sans les détails des sources dans le texte
    """
    table_info = parsed_json.get("table", {})
    table_name = table_info.get("name", "Inconnue")
    table_objective = table_info.get("objective", "Non spécifié")
    
    variables = parsed_json.get("variables", [])
    variables_str = ""
    
    for i, var in enumerate(variables, start=1):
        var_name = var.get("name", "Variable sans nom")
        var_desc = var.get("description", "Pas de description")
        var_source = var.get("source", [])
        variables_str += f'    {i}. "{var_name}" - {var_desc} {var_source}\n'
    
    return f'>> Table pertinente : "{table_name}"\n    - Objectif de la table : {table_objective}\nVariables pertinentes :\n{variables_str}'