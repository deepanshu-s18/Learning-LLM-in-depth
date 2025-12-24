"""
Process Reward Model (PRM) Guided Beam Search for Step-by-Step LLM Reasoning.

Implements inference-time compute scaling via search over reasoning steps,
scoring partial reasoning traces using a PRM / sequence classification reward model,
and visualizing the resulting reasoning tree graph.
"""

import os
import textwrap
import torch
import networkx as nx
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def get_default_device():
    """Detect available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stepwise_prm_score(prompt: str, trace: str, reward_model, tokenizer, device=None) -> float:
    """
    Simulate stepwise reward using an outcome/process reward model.
    Evaluates intermediate prefixes of reasoning steps and averages cumulative scores.
    """
    if device is None:
        device = next(reward_model.parameters()).device if hasattr(reward_model, "parameters") else "cpu"

    steps = [s.strip() for s in trace.split(". ") if s.strip()]
    if not steps:
        steps = [trace.strip()] if trace.strip() else [""]

    cumulative_score = 0.0
    for i in range(1, len(steps) + 1):
        partial = prompt + "\n" + ". ".join(steps[:i])
        inputs = tokenizer(partial, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = reward_model(**inputs)
            if hasattr(outputs, "logits"):
                if outputs.logits.numel() == 1:
                    score = outputs.logits[0].item()
                else:
                    score = outputs.logits[0][0].item()
            else:
                score = float(outputs[0])
        cumulative_score += score

    return cumulative_score / len(steps) if steps else 0.0


def beam_search_with_prm(
    prompt: str,
    reasoning_model,
    reasoning_tokenizer,
    reward_model,
    reward_tokenizer,
    is_seq2seq: bool = False,
    N: int = 4,
    M: int = 2,
    max_steps: int = 3,
    max_new_tokens: int = 64,
    device=None,
):
    """
    Performs PRM-guided Beam Search over multi-step reasoning generation.
    Supports both Seq2Seq (Flan-T5) and Causal LMs (Zephyr, TinyLlama).
    """
    assert N % M == 0, f"N ({N}) must be divisible by M ({M})"
    if device is None:
        device = next(reasoning_model.parameters()).device

    # Format prompt
    if is_seq2seq:
        formatted_prompt = (
            f"Question: {prompt}\n"
            f"Answer: Let's think step by step."
        )
    elif hasattr(reasoning_tokenizer, "chat_template") and reasoning_tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant that solves problems step-by-step."},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted_prompt = reasoning_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        formatted_prompt = f"Question: {prompt}\nAnswer: Let's think step by step.\n"

    graph = nx.DiGraph()
    beams = []

    # Step 0: Initial N completions
    input_ids = reasoning_tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    pad_token_id = reasoning_tokenizer.eos_token_id or reasoning_tokenizer.pad_token_id

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "num_return_sequences": N,
    }
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    outputs = reasoning_model.generate(**gen_kwargs)

    for i in range(N):
        gen_text = reasoning_tokenizer.decode(outputs[i], skip_special_tokens=True)
        completion = gen_text.replace(formatted_prompt, "").strip() if not is_seq2seq else gen_text.strip()
        score = stepwise_prm_score(prompt, completion, reward_model, reward_tokenizer, device=device)
        node_id = f"step0-b{i}"
        label = completion[:40] + "..." if len(completion) > 40 else (completion or "Initial")
        graph.add_node(node_id, label=label, score=score, full_text=completion, step=0)
        beams.append((completion if is_seq2seq else gen_text, score, node_id))

    # Steps 1 to max_steps: Expand top M beams with (N // M) branches each
    for step in range(1, max_steps + 1):
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:M]
        candidates = []
        branches_per_parent = N // M

        for parent_idx, (parent_text, _, parent_id) in enumerate(beams):
            if is_seq2seq:
                current_input = f"Question: {prompt}\nPartial Answer: {parent_text}\nNext reasoning step:"
            else:
                current_input = parent_text

            parent_inputs = reasoning_tokenizer(current_input, return_tensors="pt").input_ids.to(device)
            gen_step_kwargs = {
                "input_ids": parent_inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 50,
                "num_return_sequences": branches_per_parent,
            }
            if pad_token_id is not None:
                gen_step_kwargs["pad_token_id"] = pad_token_id

            children = reasoning_model.generate(**gen_step_kwargs)

            for i in range(branches_per_parent):
                child_text = reasoning_tokenizer.decode(children[i], skip_special_tokens=True)
                if is_seq2seq:
                    continuation = child_text.strip()
                    full_trace = f"{parent_text} {continuation}"
                else:
                    continuation = child_text.replace(parent_text, "").strip()
                    full_trace = child_text

                score = stepwise_prm_score(prompt, full_trace, reward_model, reward_tokenizer, device=device)
                node_id = f"step{step}-p{parent_idx}-b{i}"
                label = continuation[:40] + "..." if len(continuation) > 40 else (continuation or "Step")
                graph.add_node(node_id, label=label, score=score, full_text=continuation, step=step)
                graph.add_edge(parent_id, node_id)
                candidates.append((full_trace, score, node_id))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:N]

    return beams, graph


def plot_trace_graph_tree_clean(graph: nx.DiGraph, figsize=(14, 8), title="Beam Search Tree (PRM-Guided)", save_path=None):
    """
    Visualizes the reasoning beam search tree with node color coding based on PRM score.
    """
    if len(graph.nodes) == 0:
        print("Graph is empty, skipping plot.")
        return

    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except Exception:
        # Hierarchical layout fallback based on node step attribute
        pos = {}
        steps = {}
        for n, data in graph.nodes(data=True):
            s = data.get("step", 0)
            steps.setdefault(s, []).append(n)
        for s, nodes in steps.items():
            for idx, n in enumerate(nodes):
                pos[n] = (idx - (len(nodes) - 1) / 2.0, -float(s) * 2.0)

    scores = nx.get_node_attributes(graph, "score")
    labels = nx.get_node_attributes(graph, "label")
    node_colors = [scores.get(n, 0.0) for n in graph.nodes()]
    node_order = list(graph.nodes())

    min_val = min(node_colors) if node_colors else 0.0
    max_val = max(node_colors) if node_colors else 1.0
    val_range = max_val - min_val if max_val != min_val else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5, edge_color="#64748b", arrows=True, arrowsize=15)

    # Draw nodes
    for node in node_order:
        x, y = pos[node]
        score = scores.get(node, 0.0)
        norm_score = (score - min_val) / val_range
        fill_color = plt.cm.viridis(norm_score)

        ax.scatter(x, y, s=900, color=[fill_color], edgecolors="#1e293b", linewidths=1.5, zorder=5)

        # Label below node
        text = textwrap.shorten(labels.get(node, ""), width=35, placeholder="...")
        ax.text(
            x,
            y - 0.35,
            f"{text}\nPRM: {score:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#94a3b8", lw=0.6, alpha=0.9),
            zorder=10,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array(node_colors)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("PRM Step Score", fontsize=11, fontweight="bold")

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Tree visualization saved to: {save_path}", flush=True)

    plt.close()
