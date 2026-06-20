import os
import sys
from pathlib import Path
from datetime import datetime


def generer_arborescence(chemin, prefixe="", est_dernier=True, lignes=None):
   
    if lignes is None:
        lignes = []
    
    chemin_obj = Path(chemin)
    
    # Ajouter le nom de l'élément actuel
    if prefixe == "":
        # Racine
        lignes.append(f"{chemin_obj.name}/")
    else:
        # Élément enfant
        symbole = "└── " if est_dernier else "├── "
        lignes.append(f"{prefixe}{symbole}{chemin_obj.name}")
    
    # Si c'est un répertoire, parcourir son contenu
    if chemin_obj.is_dir():
        try:
            # Obtenir tous les éléments (dossiers et fichiers)
            elements = sorted(chemin_obj.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            
            # Calculer le nouveau préfixe pour les enfants
            extension = "    " if est_dernier else "│   "
            nouveau_prefixe = prefixe + extension
            
            # Parcourir chaque élément
            for i, element in enumerate(elements):
                est_dernier_element = (i == len(elements) - 1)
                generer_arborescence(element, nouveau_prefixe, est_dernier_element, lignes)
                
        except PermissionError:
            lignes.append(f"{prefixe}    [Permission refusée]")
        except Exception as e:
            lignes.append(f"{prefixe}    [Erreur: {e}]")
    
    return lignes


def main():
    """Fonction principale."""
    # Chemin par défaut
    chemin = "D:\\ASTA"
    
    # Récupérer le chemin depuis les arguments si fourni
    if len(sys.argv) > 1:
        chemin = sys.argv[1]
    
    # Normaliser le chemin
    chemin = os.path.normpath(chemin)
    chemin_obj = Path(chemin)
    
    # Vérifier que le chemin existe
    if not chemin_obj.exists():
        print(f"Erreur: Le chemin '{chemin}' n'existe pas.")
        sys.exit(1)
    
    # Vérifier que c'est un répertoire
    if not chemin_obj.is_dir():
        print(f"Erreur: '{chemin}' n'est pas un répertoire.")
        sys.exit(1)
    
    # Générer l'arborescence
    print(f"Génération de l'arborescence de: {chemin_obj.absolute()}")
    lignes = generer_arborescence(chemin_obj)
    
    # Créer le nom du fichier de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_fichier = f"arborescence_{chemin_obj.name}_{timestamp}.txt"
    
    # Enregistrer dans un fichier texte
    try:
        with open(nom_fichier, 'w', encoding='utf-8') as f:
            f.write(f"Arborescence de: {chemin_obj.absolute()}\n")
            f.write(f"Générée le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            for ligne in lignes:
                f.write(ligne + "\n")
        
        print(f"\n[OK] Arborescence enregistree dans: {nom_fichier}")
        print(f"  Nombre total d'elements: {len(lignes)}")
        
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
