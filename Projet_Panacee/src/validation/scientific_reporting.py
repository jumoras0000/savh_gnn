"""
Rapport scientifique automatisé pour publications académiques.

Génère des documents professionnels pour présenter les résultats :
  - Rapports LaTeX compilables
  - Documents Markdown formatés
  - Figures et tableaux
  - Justifications statistiques
  - Méthodes et résultats
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("panacee.reporting")


# ═══════════════════════════════════════════════════════════════
#  Report Structure
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScientificReport:
    """Structure d'un rapport scientifique."""
    title: str
    authors: List[str]
    abstract: str
    introduction: str
    methods: str
    results: Dict[str, Any]
    discussion: str
    conclusion: str
    references: List[str]


# ═══════════════════════════════════════════════════════════════
#  LaTeX Report Generator
# ═══════════════════════════════════════════════════════════════

class LaTeXReportGenerator:
    """
    Génère un rapport LaTeX compilable et professionnel.
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: répertoire de sortie
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        sections: Dict[str, str],
        tables: Dict[str, str],
        figures: Dict[str, Path],
        references: List[str],
    ) -> Path:
        """
        Génère un document LaTeX complet.

        Args:
            title: titre du rapport
            authors: liste des auteurs
            abstract: résumé
            sections: {section_name: section_content}
            tables: {table_name: latex_table_str}
            figures: {figure_name: path_to_image}
            references: liste des références

        Returns:
            Chemin du fichier .tex généré
        """
        latex_doc = self._create_document_header(title, authors)
        latex_doc += self._create_abstract(abstract)
        latex_doc += self._create_sections(sections)
        latex_doc += self._create_tables(tables)
        latex_doc += self._create_figures(figures)
        latex_doc += self._create_references(references)
        latex_doc += self._close_document()

        output_file = self.output_dir / f"{title.replace(' ', '_')}.tex"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_doc)

        logger.info(f"LaTeX report generated: {output_file}")
        return output_file

    def _create_document_header(self, title: str, authors: List[str]) -> str:
        """En-tête du document LaTeX."""
        authors_str = " \\and ".join(authors)
        date = datetime.now().strftime("%d %B %Y")

        return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}
\\usepackage{{listings}}

\\title{{{title}}}
\\author{{{authors_str}}}
\\date{{{date}}}

\\begin{{document}}

\\maketitle

"""

    def _create_abstract(self, abstract: str) -> str:
        """Section abstract."""
        return f"""
\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section*{{Introduction}}

"""

    def _create_sections(self, sections: Dict[str, str]) -> str:
        """Crée les sections du rapport."""
        content = ""
        for section_name, section_content in sections.items():
            # Échapper les caractères LaTeX
            safe_content = self._escape_latex(section_content)
            content += f"\\section{{{section_name}}}\n{safe_content}\n\n"
        return content

    def _create_tables(self, tables: Dict[str, str]) -> str:
        """Insère les tableaux."""
        content = ""
        for table_name, table_latex in tables.items():
            content += f"""
\\begin{{table}}[h]
\\centering
\\caption{{{table_name}}}
{table_latex}
\\end{{table}}

"""
        return content

    def _create_figures(self, figures: Dict[str, Path]) -> str:
        """Insère les figures."""
        content = ""
        for fig_name, fig_path in figures.items():
            fig_path_str = str(fig_path).replace("\\", "/")
            content += f"""
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{fig_path_str}}}
\\caption{{{fig_name}}}
\\end{{figure}}

"""
        return content

    def _create_references(self, references: List[str]) -> str:
        """Crée la section références."""
        if not references:
            return ""

        refs_str = "\n".join(
            f"\\bibitem{{{i}}} {ref}" for i, ref in enumerate(references)
        )

        return f"""
\\begin{{thebibliography}}{{99}}
{refs_str}
\\end{{thebibliography}}

"""

    def _close_document(self) -> str:
        """Ferme le document."""
        return "\\end{document}"

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Échappe les caractères spéciaux LaTeX.

        Passe UNIQUE via regex : sinon des remplacements successifs ré-échappent
        les backslashes/accolades qu'ils viennent d'introduire (LaTeX cassé).
        """
        conv = {
            "\\": r"\textbackslash{}",
            "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        }
        regex = re.compile("|".join(re.escape(k) for k in conv))
        return regex.sub(lambda m: conv[m.group()], text)


# ═══════════════════════════════════════════════════════════════
#  Markdown Report Generator
# ═══════════════════════════════════════════════════════════════

class MarkdownReportGenerator:
    """
    Génère un rapport Markdown lisible et bien formaté.
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: répertoire de sortie
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        title: str,
        sections: Dict[str, str],
        tables: Dict[str, List[Dict]],
        figures: Dict[str, Path],
        metadata: Dict[str, str] = None,
    ) -> Path:
        """
        Génère un document Markdown.

        Args:
            title: titre
            sections: {nom: contenu}
            tables: {nom: list_of_dicts}
            figures: {nom: path}
            metadata: {clé: valeur}

        Returns:
            Chemin du fichier .md
        """
        md_content = f"# {title}\n\n"

        if metadata:
            md_content += self._create_metadata_block(metadata)

        md_content += self._create_sections(sections)
        md_content += self._create_tables(tables)
        md_content += self._create_figures(figures)

        output_file = self.output_dir / f"{title.replace(' ', '_')}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Markdown report generated: {output_file}")
        return output_file

    def _create_metadata_block(self, metadata: Dict[str, str]) -> str:
        """Bloc de métadonnées YAML."""
        md = "```yaml\n"
        for key, value in metadata.items():
            md += f"{key}: {value}\n"
        md += "```\n\n"
        return md

    def _create_sections(self, sections: Dict[str, str]) -> str:
        """Crée les sections."""
        content = ""
        for section_name, section_text in sections.items():
            content += f"## {section_name}\n\n{section_text}\n\n"
        return content

    def _create_tables(self, tables: Dict[str, List[Dict]]) -> str:
        """Crée les tableaux au format Markdown."""
        content = ""
        for table_name, rows in tables.items():
            if not rows:
                continue

            content += f"### {table_name}\n\n"

            # En-tête
            headers = list(rows[0].keys())
            header_row = "| " + " | ".join(headers) + " |\n"
            separator = "|" + "|".join(["---"] * len(headers)) + "|\n"

            content += header_row + separator

            # Lignes
            for row in rows:
                values = [str(row.get(h, "N/A")) for h in headers]
                content += "| " + " | ".join(values) + " |\n"

            content += "\n"

        return content

    def _create_figures(self, figures: Dict[str, Path]) -> str:
        """Insère les figures."""
        content = ""
        for fig_name, fig_path in figures.items():
            fig_path_str = str(fig_path)
            content += f"### {fig_name}\n\n![{fig_name}]({fig_path_str})\n\n"
        return content


# ═══════════════════════════════════════════════════════════════
#  Result Summarization
# ═══════════════════════════════════════════════════════════════

class ResultSummarizer:
    """
    Résume les résultats de manière académiquement rigoureuse.
    """

    @staticmethod
    def summarize_metrics(
        metrics: Dict[str, float],
        uncertainty: Dict[str, float] = None,
    ) -> str:
        """
        Génère un résumé textuel des métriques.

        Format académique : "Metric = value ± uncertainty [confidence_interval]"
        """
        summary = ""

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                unc = uncertainty.get(metric_name, 0.0) if uncertainty else 0.0
                summary += f"- **{metric_name}**: {metric_value:.4f} ± {unc:.4f}\n"
            elif isinstance(metric_value, dict):
                summary += f"- **{metric_name}**:\n"
                for sub_name, sub_val in metric_value.items():
                    summary += f"  - {sub_name}: {sub_val:.4f}\n"

        return summary

    @staticmethod
    def generate_ablation_summary(
        ablation_results: List[Dict],
    ) -> str:
        """
        Génère un résumé des études d'ablation.
        """
        summary = "## Ablation Study Results\n\n"

        summary += "| Component | Impact (%) | Statistical Significance |\n"
        summary += "|-----------|-----------|------------------------|\n"

        for result in ablation_results:
            comp = result.get("component", "Unknown")
            impact = result.get("impact", 0.0) * 100
            sig = "Yes" if result.get("p_value", 1.0) < 0.05 else "No"
            summary += f"| {comp} | {impact:.2f}% | {sig} |\n"

        return summary

    @staticmethod
    def generate_statistical_statement(
        metric_name: str,
        mean: float,
        std: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float = None,
    ) -> str:
        """
        Génère une déclaration statistique académiquement correcte.

        Ex: "Model A achieved 85.2% accuracy (M=0.852, SD=0.032, 95% CI [0.821, 0.883])"
        """
        statement = f"{metric_name}: M={mean:.4f}, SD={std:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]"

        if p_value is not None:
            statement += f", p={'<.001' if p_value < 0.001 else f'={p_value:.4f}'}"

        return statement


# ═══════════════════════════════════════════════════════════════
#  Comparison Table Generator
# ═══════════════════════════════════════════════════════════════

class ComparisonTableGenerator:
    """
    Génère des tableaux de comparaison formatés.
    """

    @staticmethod
    def generate_model_comparison_table(
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Génère un tableau comparant les performances de plusieurs modèles.

        Args:
            results: {model_name: {metric_name: value}}

        Returns:
            Tableau LaTeX
        """
        if not results:
            return ""

        # Collecte les métriques
        metrics = set()
        for model_results in results.values():
            metrics.update(model_results.keys())

        metrics = sorted(list(metrics))

        # En-tête
        latex = "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\toprule\n"
        latex += "Model & " + " & ".join(metrics) + " \\\\\n"
        latex += "\\midrule\n"

        # Lignes
        for model_name, model_results in results.items():
            values = []
            for metric in metrics:
                val = model_results.get(metric, np.nan)
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))

            latex += model_name + " & " + " & ".join(values) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"

        return latex

    @staticmethod
    def generate_ablation_table(
        ablation_results: List[Dict],
    ) -> str:
        """
        Génère un tableau d'ablation.
        """
        latex = "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Component & Impact & p-value \\\\\n"
        latex += "\\midrule\n"

        for result in ablation_results:
            comp = result.get("component", "?")
            impact = result.get("impact", 0.0) * 100
            p_val = result.get("p_value", 1.0)

            latex += f"{comp} & {impact:.2f}\\% & {p_val:.4f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"

        return latex
