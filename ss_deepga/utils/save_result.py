
"""
save the result in memeoruy in txt file


"""
import os
import pandas as pd
import pickle

def save_deepga_run_summary_txt(results, pop, bestind, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = []
    lines.append("=== DeepGA Run Summary ===\n")

    # ---- results (DataFrame) ----
    lines.append("== results (per generation) ==\n")
    if isinstance(results, pd.DataFrame):
        lines.append(results.to_string(index=True))
    else:
        lines.append(str(results))
    lines.append("\n\n")

    # ---- bestind ----
    lines.append("== bestind (best individual) ==\n")
    try:
        enc = bestind[0]
        lines.append(f"fitness: {bestind[1]}\n")
        lines.append(f"accuracy: {bestind[2]}\n")
        lines.append(f"params: {bestind[3]}\n")
        lines.append("\n-- encoding details --\n")
        # Try to print useful attributes if present
        for attr in ["n_conv", "n_full", "first_level", "second_level"]:
            if hasattr(enc, attr):
                lines.append(f"{attr}: {getattr(enc, attr)}\n")
            else:
                lines.append(f"{attr}: <not present>\n")
    except Exception as e:
        lines.append(f"Could not format bestind: {e}\n")
        lines.append(str(bestind) + "\n")
    lines.append("\n")

    # ---- pop (top-k) ----
    lines.append("== pop (final population) ==\n")
    lines.append(f"population_size: {len(pop)}\n\n")

    # Sort by fitness descending and dump top K
    top_k = min(10, len(pop))
    pop_sorted = sorted(pop, key=lambda x: x[1], reverse=True)

    lines.append(f"-- top {top_k} individuals by fitness --\n")
    for i in range(top_k):
        ind = pop_sorted[i]
        enc = ind[0]
        lines.append(f"\n[{i+1}] fitness={ind[1]}  acc={ind[2]}  params={ind[3]}\n")
        for attr in ["n_conv", "n_full"]:
            if hasattr(enc, attr):
                lines.append(f"  {attr}: {getattr(enc, attr)}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print("Saved:", out_path)
"""
# Example usage (choose a path)
out_txt = "/content/drive/MyDrive/Workspace/Actividad_DeepGA/DeepGA_kvasir/deepga_summary_708.txt"
save_deepga_run_summary_txt(results, pop, bestind, out_txt)
"""