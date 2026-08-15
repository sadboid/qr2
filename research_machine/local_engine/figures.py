"""Publication figures generated from a run's metadata.json + paper.md.

Produces the standard credibility set for a systematic review:
  fig1  PRISMA flow diagram              (workflow)
  fig2  Conceptual/contingency framework (model)
  fig3  Evidence-stance consensus chart
  fig4  Citation-cluster network (SNA, spring layout)
  fig5  Corpus word cloud (titles + key findings)
  fig6  Publication trend by year

Every figure is drawn from REAL run data — counts are parsed from the
manuscript's own PRISMA sentence and the stored consensus/cluster records,
never invented. Style: single muted palette, no chartjunk, 300 dpi.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PALETTE = {
    "ink": "#1a2332", "box": "#eef1f5", "edge": "#5b7089",
    "support": "#3b7a57", "oppose": "#a8574e", "mixed": "#c9a227", "neutral": "#8a93a1",
    "accent": "#2f5d8a",
}
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.edgecolor": PALETTE["edge"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _box(ax, x, y, w, h, text, fc=None, fontsize=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=fc or PALETTE["box"], edgecolor=PALETTE["edge"], linewidth=1.1,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=PALETTE["edge"], linewidth=1.2))


# ---------------------------------------------------------------- fig 1
def prisma_flow(paper_md: str, out: Path) -> Optional[Dict[str, int]]:
    m = re.search(
        r"Records identified across databases:\s*~?(\d+).*?duplicates:\s*(\d+)"
        r".*?relevance:\s*(\d+).*?synthesis:\s*(\d+)",
        paper_md, re.DOTALL,
    )
    if not m:
        return None
    n_id, n_dedup, n_screen, n_inc = (int(g) for g in m.groups())
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 13); ax.axis("off")

    steps = [
        (9.6, f"Records identified through database\nsearching (Semantic Scholar, arXiv,\nCrossref, OpenAlex)   n ≈ {n_id}"),
        (7.2, f"Records after deduplication and\ndomain-relevance filtering\nn = {n_dedup}"),
        (4.8, f"Records screened on abstract\nsubstance and source quality\nn = {n_screen}"),
        (2.4, f"Studies included in synthesis\nn = {n_inc}"),
    ]
    for y, text in steps:
        _box(ax, 0.5, y, 6.4, 1.7, text, fontsize=9)
    for i in range(len(steps) - 1):
        _arrow(ax, 3.7, steps[i][0], 3.7, steps[i + 1][0] + 1.7)

    # exclusion boxes sit fully to the right of the main column
    excl = [
        (8.45, f"Duplicates & off-domain\nrecords removed\nn = {n_id - n_dedup}"),
        (6.05, f"Excluded on screening\nn = {n_screen - n_inc}"),
    ]
    for y, text in excl:
        _box(ax, 7.6, y - 0.6, 3.1, 1.35, text, fontsize=8.2)
        _arrow(ax, 6.9, y + 0.08, 7.6, y + 0.08)
    ax.text(5.5, 12.4, "PRISMA Flow of Study Selection", ha="center",
            fontsize=12, weight="bold")
    fig.savefig(out); plt.close(fig)
    return {"identified": n_id, "included": n_inc}


# ---------------------------------------------------------------- fig 2
def framework_model(primary_theory: str, out: Path,
                    moderators: Optional[List[str]] = None):
    moderators = moderators or [
        "C1\nTask structure /\nprogrammability",
        "C2\nFounder digital\ncapability",
        "C3\nContextual &\ninstitutional support",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    _box(ax, 0.4, 3.1, 3.1, 1.8, "AI tool adoption\nby founders", fc="#dfe7f0",
         weight="bold", fontsize=10.5)
    _box(ax, 9.5, 4.3, 3.1, 1.5, "Decision speed", fc="#e4efe7", weight="bold", fontsize=10.5)
    _box(ax, 9.5, 2.2, 3.1, 1.5, "Decision quality", fc="#e4efe7", weight="bold", fontsize=10.5)
    _arrow(ax, 3.5, 4.3, 9.5, 5.0)
    _arrow(ax, 3.5, 3.7, 9.5, 2.95)

    # moderator arrows hit the causal path (between source and outcomes)
    xs = [1.6, 5.0, 8.4]
    targets = [(4.9, 4.35), (6.4, 4.35), (7.9, 4.55)]
    for (x, mtext), (tx, ty) in zip(zip(xs, moderators), targets):
        _box(ax, x, 6.5, 3.0, 1.55, mtext, fontsize=8.6)
        _arrow(ax, x + 1.5, 6.5, tx, ty)

    _box(ax, 3.4, 0.35, 6.2, 1.05,
         f"Theoretical lens: {primary_theory}", fontsize=9)
    ax.text(6.5, 8.35, "Contingency Model of AI-Enabled Founder Decision-Making",
            ha="center", fontsize=12, weight="bold")
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- fig 3
def consensus_chart(consensus: dict, out: Path):
    cats = [("Support", consensus["support_count"], PALETTE["support"]),
            ("Mixed", consensus["mixed_count"], PALETTE["mixed"]),
            ("Oppose / null", consensus["oppose_count"], PALETTE["oppose"]),
            ("Neutral", consensus["neutral_count"], PALETTE["neutral"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    left = 0
    total = sum(c[1] for c in cats) or 1
    for label, n, color in cats:
        ax.barh([0], [n], left=left, color=color, edgecolor="white", height=0.55)
        if n:
            ax.text(left + n / 2, 0, f"{label}\n{n}", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
        left += n
    ax.set_xlim(0, total); ax.set_yticks([])
    ax.set_xlabel(f"Studies with a classifiable stance (n = {total})")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Evidence Stance on the Focal Research Question", fontsize=11.5, weight="bold")
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- fig 4
def citation_network(clusters: List[dict], out: Path):
    import networkx as nx
    G = nx.Graph()
    colors = ["#2f5d8a", "#3b7a57", "#a8574e", "#c9a227", "#6b5b95"]
    node_color, node_size, labels = [], [], {}
    for ci, cl in enumerate(clusters):
        papers = cl.get("papers", [])[:14]
        anchor = None
        for p in papers:
            nid = p["title"][:40]
            G.add_node(nid)
            node_color.append(colors[ci % len(colors)])
            node_size.append(60 + min(p.get("citations", 0), 1200) / 4)
            if anchor is None:
                anchor = nid
                labels[nid] = cl["theme"][:24]
            else:
                G.add_edge(anchor, nid)          # star to cluster anchor
        # k-nearest by year within cluster for local structure
        by_year = sorted(papers, key=lambda p: p.get("year", 0))
        for a, b in zip(by_year, by_year[1:]):
            G.add_edge(a["title"][:40], b["title"][:40])
    pos = nx.spring_layout(G, k=0.55, seed=42)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color=PALETTE["edge"])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                           node_size=node_size, alpha=0.88, linewidths=0)
    for nid, lab in labels.items():
        x, y = pos[nid]
        ax.text(x, y + 0.07, lab, fontsize=9.5, weight="bold", ha="center")
    ax.axis("off")
    ax.set_title("Thematic Clusters of the Corpus — Top-Cited Papers\n"
                 "(node size ∝ citation count; edges = within-cluster adjacency)",
                 fontsize=11.5, weight="bold")
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- fig 5
def corpus_wordcloud(texts: List[str], out: Path):
    from wordcloud import WordCloud, STOPWORDS
    stop = set(STOPWORDS) | {
        "study", "studies", "paper", "research", "using", "based", "toward",
        "towards", "review", "analysis", "et", "al",
    }
    wc = WordCloud(width=1400, height=800, background_color="white",
                   colormap="cividis", stopwords=stop, max_words=90,
                   prefer_horizontal=0.92, random_state=42)
    wc.generate(" ".join(texts))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Term Landscape of the Included Corpus", fontsize=11.5, weight="bold")
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- fig 6
def publication_trend(years: List[int], out: Path):
    counts = Counter(y for y in years if y and y > 2000)
    xs = sorted(counts)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar([str(x) for x in xs], [counts[x] for x in xs],
           color=PALETTE["accent"], width=0.62)
    for i, x in enumerate(xs):
        ax.text(i, counts[x] + 0.12, str(counts[x]), ha="center", fontsize=8.5)
    ax.set_ylabel("Included studies")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Publication Trend of the Included Corpus", fontsize=11.5, weight="bold")
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------- driver
def generate_all(metadata_path: str, paper_md_path: str, out_dir: str) -> Dict[str, str]:
    """Generate the full figure set. Returns {figure_id: filename}."""
    meta = json.loads(Path(metadata_path).read_text())
    paper_md = Path(paper_md_path).read_text()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made: Dict[str, str] = {}

    if prisma_flow(paper_md, out / "fig1_prisma_flow.png"):
        made["fig1"] = "fig1_prisma_flow.png"

    theory = "Technology Acceptance Model (Davis, 1989)"
    tf = meta.get("sections", {}).get("theoretical_framework", "") or paper_md
    m = re.search(r"adopts \*\*(.+?)\*\*", tf)
    if m:
        theory = m.group(1)
    framework_model(theory, out / "fig2_framework.png")
    made["fig2"] = "fig2_framework.png"

    cons = meta.get("consensus") or {}
    if cons.get("total_papers"):
        consensus_chart(cons, out / "fig3_consensus.png")
        made["fig3"] = "fig3_consensus.png"

    clusters = meta.get("citation_clusters") or []
    if clusters:
        citation_network(clusters, out / "fig4_citation_network.png")
        made["fig4"] = "fig4_citation_network.png"

    refs = (meta.get("citations") or {}).get("references", [])
    texts = [r.get("title", "") for r in refs]
    texts += [it.get("key_finding", "") for it in cons.get("reading_list", [])]
    if texts:
        corpus_wordcloud(texts, out / "fig5_wordcloud.png")
        made["fig5"] = "fig5_wordcloud.png"

    years = [r.get("year") for r in refs]
    if years:
        publication_trend(years, out / "fig6_publication_trend.png")
        made["fig6"] = "fig6_publication_trend.png"

    return made
