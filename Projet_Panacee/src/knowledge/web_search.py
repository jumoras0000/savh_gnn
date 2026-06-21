"""
Module de recherche web pour vérification d'hypothèses.

Intègre les APIs publiques :
  - PubChem : propriétés chimiques, activité biologique
  - ChEMBL  : données pharmacologiques, cibles, essais cliniques
  - PubMed  : littérature scientifique (articles, méta-analyses)

Toutes les requêtes sont en lecture seule et respectent les rate-limits.
"""
import json
import logging
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.error_handler import safe_execution

logger = logging.getLogger("panacee.web_search")

# Rate limiting global
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.3  # secondes entre requêtes


def _rate_limited_request(url: str, timeout: int = 15) -> Optional[str]:
    """
    Effectue une requête HTTP GET avec rate limiting.

    Args:
        url: URL à requêter
        timeout: timeout en secondes

    Returns:
        Contenu de la réponse ou None
    """
    global _last_request_time

    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    _last_request_time = time.time()

    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Panacee-DrugDiscovery/1.0 (research)"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} pour {url}")
        return None
    except Exception as e:
        logger.warning(f"Requête échouée ({url[:80]}...): {e}")
        return None


# ─────────────────────────────────────────────
#  PubChem API
# ─────────────────────────────────────────────

@dataclass
class PubChemCompound:
    """Données d'un composé PubChem."""
    cid: int = 0
    iupac_name: str = ""
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    canonical_smiles: str = ""
    xlogp: Optional[float] = None
    hbond_donor: int = 0
    hbond_acceptor: int = 0
    tpsa: float = 0.0
    complexity: float = 0.0


class PubChemSearch:
    """Interface avec l'API PubChem (REST)."""

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=None, catch=(Exception,))
    def search_by_smiles(smiles: str) -> Optional[PubChemCompound]:
        """
        Recherche un composé par SMILES.

        Args:
            smiles: SMILES de la molécule

        Returns:
            PubChemCompound ou None
        """
        encoded = urllib.parse.quote(smiles, safe="")
        url = (
            f"{PubChemSearch.BASE_URL}/compound/smiles/{encoded}"
            f"/property/IUPACName,MolecularFormula,MolecularWeight,"
            f"CanonicalSMILES,XLogP,HBondDonorCount,HBondAcceptorCount,"
            f"TPSA,Complexity/JSON"
        )
        data = _rate_limited_request(url)
        if data is None:
            return None

        result = json.loads(data)
        props = result.get("PropertyTable", {}).get("Properties", [{}])[0]

        return PubChemCompound(
            cid=props.get("CID", 0),
            iupac_name=props.get("IUPACName", ""),
            molecular_formula=props.get("MolecularFormula", ""),
            molecular_weight=props.get("MolecularWeight", 0.0),
            canonical_smiles=props.get("CanonicalSMILES", ""),
            xlogp=props.get("XLogP"),
            hbond_donor=props.get("HBondDonorCount", 0),
            hbond_acceptor=props.get("HBondAcceptorCount", 0),
            tpsa=props.get("TPSA", 0.0),
            complexity=props.get("Complexity", 0.0),
        )

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=None, catch=(Exception,))
    def get_bioactivity(cid: int, max_results: int = 10) -> List[Dict]:
        """
        Récupère les données de bioactivité pour un composé.

        Args:
            cid: PubChem Compound ID
            max_results: nombre max de résultats

        Returns:
            Liste d'activités biologiques
        """
        url = (
            f"{PubChemSearch.BASE_URL}/compound/cid/{cid}"
            f"/assaysummary/JSON"
        )
        data = _rate_limited_request(url, timeout=20)
        if data is None:
            return []

        result = json.loads(data)
        summaries = result.get("Table", {}).get("Row", [])

        activities = []
        for row in summaries[:max_results]:
            cells = row.get("Cell", [])
            if len(cells) >= 5:
                activities.append({
                    "aid": cells[0],
                    "activity_outcome": cells[2] if len(cells) > 2 else "",
                    "target_name": cells[4] if len(cells) > 4 else "",
                })

        return activities

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=[], catch=(Exception,))
    def search_similar(smiles: str, threshold: int = 90, max_results: int = 5) -> List[Dict]:
        """
        Recherche des composés similaires (par similarité Tanimoto).

        Args:
            smiles: SMILES de la molécule de référence
            threshold: seuil de similarité (0-100)
            max_results: nombre max de résultats

        Returns:
            Liste de composés similaires avec leurs propriétés
        """
        encoded = urllib.parse.quote(smiles, safe="")

        # Similarity search is async on PubChem - submit then poll
        submit_url = (
            f"{PubChemSearch.BASE_URL}/compound/similarity/smiles/{encoded}"
            f"/cids/JSON?Threshold={threshold}&MaxRecords={max_results}"
        )
        data = _rate_limited_request(submit_url, timeout=20)
        if data is None:
            return []

        result = json.loads(data)

        # Check for ListKey (async)
        waiting = result.get("Waiting", {})
        if waiting:
            list_key = waiting.get("ListKey")
            if list_key:
                # Poll for results
                for _ in range(5):
                    time.sleep(2)
                    poll_url = (
                        f"{PubChemSearch.BASE_URL}/compound/listkey/{list_key}"
                        f"/property/IUPACName,MolecularWeight,CanonicalSMILES,XLogP"
                        f"/JSON"
                    )
                    data = _rate_limited_request(poll_url)
                    if data:
                        result = json.loads(data)
                        if "PropertyTable" in result:
                            break

        props_list = result.get("PropertyTable", {}).get("Properties", [])
        return [
            {
                "cid": p.get("CID", 0),
                "name": p.get("IUPACName", ""),
                "mw": p.get("MolecularWeight", 0),
                "smiles": p.get("CanonicalSMILES", ""),
                "logp": p.get("XLogP"),
            }
            for p in props_list[:max_results]
        ]


# ─────────────────────────────────────────────
#  ChEMBL API
# ─────────────────────────────────────────────

class ChEMBLSearch:
    """Interface avec l'API ChEMBL."""

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=None, catch=(Exception,))
    def search_molecule(smiles: str) -> Optional[Dict]:
        """
        Recherche une molécule dans ChEMBL par SMILES.

        Returns:
            Données ChEMBL ou None
        """
        encoded = urllib.parse.quote(smiles, safe="")
        url = (
            f"{ChEMBLSearch.BASE_URL}/molecule.json"
            f"?molecule_structures__canonical_smiles={encoded}"
        )
        data = _rate_limited_request(url)
        if data is None:
            return None

        result = json.loads(data)
        molecules = result.get("molecules", [])
        if not molecules:
            return None

        mol = molecules[0]
        return {
            "chembl_id": mol.get("molecule_chembl_id", ""),
            "pref_name": mol.get("pref_name", ""),
            "molecule_type": mol.get("molecule_type", ""),
            "max_phase": mol.get("max_phase", 0),
            "oral": mol.get("oral", False),
            "parenteral": mol.get("parenteral", False),
            "natural_product": mol.get("natural_product", -1),
        }

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=[], catch=(Exception,))
    def get_activities(chembl_id: str, max_results: int = 20) -> List[Dict]:
        """
        Récupère les activités biologiques d'une molécule ChEMBL.

        Args:
            chembl_id: identifiant ChEMBL (ex: CHEMBL25)
            max_results: nombre max de résultats

        Returns:
            Liste d'activités
        """
        url = (
            f"{ChEMBLSearch.BASE_URL}/activity.json"
            f"?molecule_chembl_id={chembl_id}&limit={max_results}"
        )
        data = _rate_limited_request(url, timeout=20)
        if data is None:
            return []

        result = json.loads(data)
        activities = result.get("activities", [])

        return [
            {
                "target_name": a.get("target_pref_name", ""),
                "target_type": a.get("target_type", ""),
                "activity_type": a.get("standard_type", ""),
                "activity_value": a.get("standard_value"),
                "activity_units": a.get("standard_units", ""),
                "assay_type": a.get("assay_type", ""),
            }
            for a in activities
        ]

    @staticmethod
    @safe_execution(retries=2, delay=1.0, fallback=[], catch=(Exception,))
    def search_target(query: str, max_results: int = 5) -> List[Dict]:
        """
        Recherche des cibles thérapeutiques.

        Args:
            query: terme de recherche (ex: "HIV protease")
            max_results: nombre max de résultats

        Returns:
            Liste de cibles
        """
        encoded = urllib.parse.quote(query)
        url = (
            f"{ChEMBLSearch.BASE_URL}/target/search.json"
            f"?q={encoded}&limit={max_results}"
        )
        data = _rate_limited_request(url)
        if data is None:
            return []

        result = json.loads(data)
        targets = result.get("targets", [])

        return [
            {
                "chembl_id": t.get("target_chembl_id", ""),
                "pref_name": t.get("pref_name", ""),
                "target_type": t.get("target_type", ""),
                "organism": t.get("organism", ""),
            }
            for t in targets
        ]


# ─────────────────────────────────────────────
#  PubMed Search (E-utilities)
# ─────────────────────────────────────────────

@dataclass
class PubMedArticle:
    """Article PubMed simplifié."""
    pmid: str = ""
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    abstract_snippet: str = ""


class PubMedSearch:
    """Interface avec l'API PubMed E-utilities (NCBI)."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    @staticmethod
    @safe_execution(retries=2, delay=2.0, fallback=[], catch=(Exception,))
    def search_articles(
        query: str, max_results: int = 5
    ) -> List[PubMedArticle]:
        """
        Recherche d'articles dans PubMed.

        Args:
            query: requête de recherche
            max_results: nombre max d'articles

        Returns:
            Liste d'articles PubMed
        """
        # Étape 1 : recherche des IDs
        encoded = urllib.parse.quote(query)
        search_url = (
            f"{PubMedSearch.ESEARCH_URL}?db=pubmed&term={encoded}"
            f"&retmax={max_results}&retmode=json&sort=relevance"
        )
        data = _rate_limited_request(search_url)
        if data is None:
            return []

        search_result = json.loads(data)
        id_list = search_result.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        # Étape 2 : récupérer les résumés
        ids = ",".join(id_list)
        summary_url = (
            f"{PubMedSearch.ESUMMARY_URL}?db=pubmed&id={ids}&retmode=json"
        )
        data = _rate_limited_request(summary_url, timeout=20)
        if data is None:
            return []

        summary_result = json.loads(data)
        articles = []

        result_data = summary_result.get("result", {})
        for pmid in id_list:
            info = result_data.get(pmid, {})
            if not info or pmid == "uids":
                continue

            # Extraire les auteurs
            authors_list = info.get("authors", [])
            authors_str = ", ".join(
                a.get("name", "") for a in authors_list[:3]
            )
            if len(authors_list) > 3:
                authors_str += " et al."

            articles.append(PubMedArticle(
                pmid=pmid,
                title=info.get("title", ""),
                authors=authors_str,
                journal=info.get("fulljournalname", info.get("source", "")),
                year=info.get("pubdate", "")[:4],
                abstract_snippet="",  # esummary ne donne pas l'abstract
            ))

        return articles


# ─────────────────────────────────────────────
#  Orchestrateur de recherche
# ─────────────────────────────────────────────

class WebResearchEngine:
    """
    Moteur de recherche unifié pour vérification d'hypothèses.

    Combine PubChem, ChEMBL et PubMed pour fournir un
    dossier complet sur une molécule ou une combinaison.
    """

    def __init__(self):
        self.pubchem = PubChemSearch()
        self.chembl = ChEMBLSearch()
        self.pubmed = PubMedSearch()
        self._cache: Dict[str, Any] = {}

    def research_molecule(self, smiles: str) -> Dict:
        """
        Recherche complète sur une molécule.

        Args:
            smiles: SMILES de la molécule

        Returns:
            Dossier de recherche complet
        """
        if smiles in self._cache:
            logger.info(f"Cache hit pour {smiles[:30]}...")
            return self._cache[smiles]

        dossier = {"smiles": smiles, "sources": []}
        logger.info(f"Recherche web pour: {smiles[:50]}...")

        # PubChem
        pc = PubChemSearch.search_by_smiles(smiles)
        if pc:
            dossier["pubchem"] = {
                "cid": pc.cid,
                "name": pc.iupac_name,
                "formula": pc.molecular_formula,
                "mw": pc.molecular_weight,
                "logp": pc.xlogp,
            }
            dossier["sources"].append("PubChem")

            # Bioactivité disponible via CID
            if pc.cid:
                activities = PubChemSearch.get_bioactivity(pc.cid, max_results=5)
                dossier["pubchem_bioactivity"] = activities

        # ChEMBL
        chembl_data = ChEMBLSearch.search_molecule(smiles)
        if chembl_data:
            dossier["chembl"] = chembl_data
            dossier["sources"].append("ChEMBL")

            # Activités ChEMBL
            if chembl_data.get("chembl_id"):
                activities = ChEMBLSearch.get_activities(
                    chembl_data["chembl_id"], max_results=10
                )
                dossier["chembl_activities"] = activities

        # PubMed - chercher la littérature
        name = ""
        if pc and pc.iupac_name:
            name = pc.iupac_name
        elif chembl_data and chembl_data.get("pref_name"):
            name = chembl_data["pref_name"]

        if name:
            articles = PubMedSearch.search_articles(
                f"{name} pharmacology", max_results=3
            )
            dossier["pubmed_articles"] = [
                {"pmid": a.pmid, "title": a.title, "authors": a.authors,
                 "journal": a.journal, "year": a.year}
                for a in articles
            ]
            dossier["sources"].append("PubMed")

        self._cache[smiles] = dossier
        return dossier

    def research_combination(
        self, smiles_list: List[str], indication: str = ""
    ) -> Dict:
        """
        Recherche sur une combinaison de molécules.

        Args:
            smiles_list: liste de SMILES
            indication: indication thérapeutique visée

        Returns:
            Dossier de recherche combinée
        """
        combo_report = {
            "molecules": [],
            "literature": [],
            "indication": indication,
        }

        # Rechercher chaque molécule
        for smiles in smiles_list:
            mol_data = self.research_molecule(smiles)
            combo_report["molecules"].append(mol_data)

        # Chercher la littérature sur la combinaison
        if indication:
            query = f"drug combination {indication}"
            articles = PubMedSearch.search_articles(query, max_results=3)
            combo_report["literature"] = [
                {"pmid": a.pmid, "title": a.title, "year": a.year}
                for a in articles
            ]

        return combo_report

    def verify_hypothesis(self, hypothesis: str) -> Dict:
        """
        Vérifie une hypothèse scientifique via PubMed.

        Args:
            hypothesis: hypothèse à vérifier (texte libre)

        Returns:
            Résultats de vérification avec articles de support
        """
        articles = PubMedSearch.search_articles(hypothesis, max_results=5)

        return {
            "hypothesis": hypothesis,
            "articles_found": len(articles),
            "articles": [
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "authors": a.authors,
                    "journal": a.journal,
                    "year": a.year,
                }
                for a in articles
            ],
            "evidence_level": (
                "strong" if len(articles) >= 4 else
                "moderate" if len(articles) >= 2 else
                "weak" if len(articles) >= 1 else
                "none"
            ),
        }

    def clear_cache(self):
        """Vide le cache de recherche."""
        self._cache.clear()
