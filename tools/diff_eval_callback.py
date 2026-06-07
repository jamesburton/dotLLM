"""Parse llama-eval-callback output and compare against dotLLM TensorDump dumps.

Maps llama.cpp tensor names to dotLLM tensor names and reports the FIRST tensor
where first-3 elements or sum disagree by more than tolerance.

Usage:
    python diff_eval_callback.py <eval_callback.txt> <dotllm_dump_dir>
"""
import sys, re, struct
from pathlib import Path
import numpy as np


# Map llama.cpp tensor names to dotLLM tensor name patterns.
# llama.cpp suffix "-L" maps to "blk.L" in dotLLM.
# Pattern: (llama_name_pattern, dotllm_name_template)
# llama_name_pattern is a regex with group "L" for layer index.
NAME_MAP = [
    # Token embedding (no layer suffix in dotLLM, single shape (1, hidden) per token in dotLLM)
    (r"^node_0$", "token_embd"),  # token_embd RESHAPE - won't match cleanly though
    # Layer-level tensors
    (r"^attn_norm-(?P<L>\d+)$", "blk.{L}.attn_norm"),
    (r"^linear_attn_qkv_mixed-(?P<L>\d+)$", "blk.{L}.linear_attn_qkv_mixed"),
    (r"^z-(?P<L>\d+)$", "blk.{L}.z"),
    (r"^alpha-(?P<L>\d+)$", "blk.{L}.alpha_proj"),
    (r"^a_softplus-(?P<L>\d+)$", None),
    (r"^gate-(?P<L>\d+)$", "blk.{L}.g"),  # gate is the GDN decay g
    (r"^beta-(?P<L>\d+)$", "blk.{L}.beta_sigmoid"),  # post-sigmoid beta
    (r"^conv_output_raw-(?P<L>\d+)$", None),  # pre-silu
    (r"^conv_output_silu-(?P<L>\d+)$", "blk.{L}.conv_output_silu"),
    (r"^q_conv-(?P<L>\d+)$", "blk.{L}.q_conv"),
    (r"^k_conv-(?P<L>\d+)$", "blk.{L}.k_conv"),
    (r"^v_conv-(?P<L>\d+)$", "blk.{L}.v_conv"),
    (r"^q_conv_predelta-(?P<L>\d+)$", "blk.{L}.q_conv_predelta"),
    (r"^k_conv_predelta-(?P<L>\d+)$", "blk.{L}.k_conv_predelta"),
    (r"^attn_output-(?P<L>\d+)$", "blk.{L}.attn_output"),
    (r"^final_output-(?P<L>\d+)$", "blk.{L}.final_output"),
    (r"^linear_attn_out-(?P<L>\d+)$", "blk.{L}.linear_attn_out"),
    (r"^attn_post_norm-(?P<L>\d+)$", "blk.{L}.attn_post_norm"),
    (r"^l_out-(?P<L>\d+)$", "blk.{L}.l_out"),
    (r"^ffn_out-(?P<L>\d+)$", "blk.{L}.ffn_out"),
    # Full-attn layer tensors
    (r"^Qcur_full-(?P<L>\d+)$", "blk.{L}.fa_qg"),
    (r"^Qcur-(?P<L>\d+)$", "blk.{L}.fa_q_postrope"),
    (r"^Kcur-(?P<L>\d+)$", "blk.{L}.fa_k_postrope"),
    (r"^Vcur-(?P<L>\d+)$", "blk.{L}.fa_v"),
    (r"^attn_pregate-(?P<L>\d+)$", "blk.{L}.fa_attnout_pregate"),
    (r"^attn_gated-(?P<L>\d+)$", "blk.{L}.fa_attnout_postgate"),
    # Final
    (r"^result_norm$", "result_norm"),
    (r"^result_output$", "result_output"),
]


def map_name(llama_name):
    """Map llama.cpp tensor name to dotLLM dump filename component, or None."""
    for pat, tpl in NAME_MAP:
        m = re.match(pat, llama_name)
        if m:
            if tpl is None:
                return None
            return tpl.format(**m.groupdict())
    return None


def parse_eval_callback(path):
    """Yield (tensor_name, shape, first3, last3, sum_value) from eval-callback output."""
    tensor_re = re.compile(r"^common_debug_cb_eval:\s+(\S+)\s+=\s+\((\w+)\)\s+(\w+)\((.+?)\)\s+=\s+\{([^}]+)\}\s*$")
    val_re = re.compile(r"\[(.+?)\]")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m = tensor_re.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        dtype = m.group(2)
        op = m.group(3)
        shape_str = m.group(5)
        shape = [int(x.strip()) for x in shape_str.split(",")]
        # Look for sum=, first3/last3 in following lines
        first_vals = None
        last_vals = None
        sum_val = None
        j = i + 1
        # Skip to "sum = " line, collecting first innermost row
        while j < len(lines) and not lines[j].startswith("common_debug_cb_eval:"):
            if "sum = " in lines[j]:
                try:
                    sum_val = float(lines[j].split("sum = ")[1].strip())
                except (ValueError, IndexError):
                    pass
                break
            # First innermost row: look for line with multiple floats
            stripped = lines[j].strip()
            if stripped.startswith("[") and ("," in stripped) and first_vals is None:
                # Extract floats
                row_str = stripped.strip("[],")
                parts = [p.strip() for p in row_str.split(",")]
                floats = []
                trailing = False
                for p in parts:
                    if p == "...":
                        trailing = True
                        continue
                    try:
                        floats.append(float(p))
                    except ValueError:
                        pass
                if floats:
                    if trailing:
                        # Format: first3..., last3
                        first_vals = floats[:3]
                        last_vals = floats[-3:]
                    else:
                        first_vals = floats[:3]
                        if len(floats) >= 3:
                            last_vals = floats[-3:]
            j += 1
        yield name, dtype, op, shape, first_vals, last_vals, sum_val
        i = j


def read_dotllm_bin(path):
    with open(path, "rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape) if shape else data


def find_dotllm_dump(dump_dir, name):
    """Find dotLLM dump file matching the given tensor name (without index prefix)."""
    safe = name.replace("/", "_").replace("\\", "_")
    matches = sorted(dump_dir.glob(f"*_{safe}.bin"))
    if not matches:
        return None
    return matches[0]


def main():
    eval_path = Path(sys.argv[1])
    dump_dir = Path(sys.argv[2])

    diverged_count = 0
    matched_count = 0
    skipped_count = 0
    first_divergence = None

    print(f"Parsing {eval_path}...")
    for name, dtype, op, shape, first3, last3, sum_val in parse_eval_callback(eval_path):
        dotllm_name = map_name(name)
        if dotllm_name is None:
            skipped_count += 1
            continue
        dump_path = find_dotllm_dump(dump_dir, dotllm_name)
        if dump_path is None:
            print(f"  [MISS] llama={name} -> dotLLM={dotllm_name} : no dump file")
            continue
        data = read_dotllm_bin(dump_path)
        flat = data.flatten()
        d_first3 = [float(flat[i]) for i in range(min(3, flat.size))]
        d_last3 = [float(flat[i]) for i in range(max(0, flat.size - 3), flat.size)]
        d_sum = float(flat.astype(np.float64).sum())

        # Match check
        ok = True
        if first3 is not None and len(first3) >= 3:
            for a, b in zip(first3, d_first3):
                if abs(a - b) > 0.01 and abs(a - b) / (abs(a) + 1e-6) > 0.005:
                    ok = False
                    break
        if ok and sum_val is not None:
            if abs(sum_val - d_sum) > 0.5 + 0.005 * abs(sum_val):
                ok = False
        if ok:
            matched_count += 1
            # print(f"  [OK]  {name} -> {dotllm_name}")
        else:
            diverged_count += 1
            if first_divergence is None:
                first_divergence = name
            print(f"  [DIFF] {name} -> {dotllm_name} shape={shape}")
            print(f"         llama first3={first3} sum={sum_val:.4f}" if sum_val is not None else f"         llama first3={first3}")
            print(f"         dotLLM first3={d_first3} sum={d_sum:.4f}")
            if diverged_count >= 5:
                print(f"  ... stopping after {diverged_count} divergences")
                break

    print()
    print(f"=== SUMMARY ===")
    print(f"  Matched: {matched_count}")
    print(f"  Diverged: {diverged_count}")
    print(f"  Skipped (unmapped): {skipped_count}")
    if first_divergence:
        print(f"  FIRST DIVERGENCE: {first_divergence}")


if __name__ == "__main__":
    main()
