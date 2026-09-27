"""Matplotlib figures for the confusion matrix and small parse trees."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tree import Tree


def confusion_figure(matrix: list[list[int]], labels: list[str]):
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
        linewidths=0.4,
        linecolor="#f4f0e6",
    )
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("Confusion matrix on the held-out test set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def tree_figure(tree: Tree | None, title: str):
    """Draw a small tree. Wide chunk trees stay as text in the interface."""
    if tree is None or not isinstance(tree, Tree):
        return None
    if len(tree.leaves()) > 16:
        return None

    nodes: list[dict] = []
    edges: list[tuple[int, int]] = []

    def leaf_label(node) -> str:
        if isinstance(node, tuple) and len(node) == 2:
            return f"{node[0]}/{node[1]}"
        return str(node)

    def walk(node, depth: int) -> int:
        index = len(nodes)
        label = node.label() if isinstance(node, Tree) else leaf_label(node)
        nodes.append({"label": label, "depth": depth, "children": [], "leaf": not isinstance(node, Tree)})
        if isinstance(node, Tree):
            for child in node:
                child_index = walk(child, depth + 1)
                nodes[index]["children"].append(child_index)
                edges.append((index, child_index))
        return index

    walk(tree, 0)

    leaf_index = 0
    for node in nodes:
        if node["leaf"]:
            node["x"] = leaf_index
            leaf_index += 1

    def assign(index: int) -> float:
        node = nodes[index]
        if "x" in node:
            return node["x"]
        xs = [assign(child) for child in node["children"]]
        node["x"] = sum(xs) / len(xs)
        return node["x"]

    assign(0)
    depth = max(node["depth"] for node in nodes)
    width = max(6.2, leaf_index * 0.85)
    height = max(2.8, (depth + 1) * 0.85)
    fig, ax = plt.subplots(figsize=(width, height))

    for parent, child in edges:
        ax.plot(
            [nodes[parent]["x"], nodes[child]["x"]],
            [-nodes[parent]["depth"], -nodes[child]["depth"]],
            color="#c4b8a5",
            linewidth=1.1,
            zorder=1,
        )

    for node in nodes:
        color = "#fffdf8" if node["leaf"] else "#1c6b58"
        font = "#1c1915" if node["leaf"] else "#fffdf8"
        ax.text(
            node["x"],
            -node["depth"],
            node["label"],
            ha="center",
            va="center",
            fontsize=9,
            color=font,
            zorder=2,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": color, "edgecolor": "#1c6b58", "linewidth": 0.8},
        )

    ax.set_title(title, fontsize=11, color="#1c1915", pad=8)
    ax.set_axis_off()
    ax.margins(0.15)
    fig.tight_layout()
    return fig
