import json
import os
import sys


def run_notebook(filepath, mock_setup=None):
    edu_dir = os.path.dirname(os.path.abspath(__file__))
    if edu_dir not in sys.path:
        sys.path.insert(0, edu_dir)

    print("\n==========================================")
    print(f"Testing Notebook: {os.path.basename(filepath)}")
    print("==========================================")
    with open(filepath, encoding="utf-8") as f:
        nb = json.load(f)

    # Extract code cells
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]

    # Build complete python source code
    source_lines = []
    for cell in code_cells:
        # Join lines of the cell
        cell_source = "".join(cell["source"])
        # Comment out colab specific pip installs/uninstalls and wget calls to avoid
        # running them locally; insert 'pass' to maintain valid python indentation blocks
        cell_source = cell_source.replace("!pip install", "pass # !pip install")
        cell_source = cell_source.replace("!pip uninstall", "pass # !pip uninstall")
        cell_source = cell_source.replace("!wget", "pass # !wget")
        source_lines.append(cell_source)

    full_source = "\n\n# --- NEW CELL ---\n\n".join(source_lines)

    # Global context for exec
    global_context = {}

    # Apply mocks if provided
    if mock_setup:
        mock_setup(global_context)

    try:
        # Execute the python script
        exec(full_source, global_context)
        print("Result: SUCCESS")
        return True
    except Exception:
        import traceback

        print("Result: FAILED", file=sys.stderr)
        traceback.print_exc()
        return False


# Mock setups to make tests run instantly
def mock_stage1_a(ctx):
    # Mock clean_and_collect to only fetch 2000 chars instead of 100000
    # Also mock epochs to 1 for super fast training
    ctx["sys"] = sys
    # Add a mock class/function wrapper
    import plotly.graph_objects as go

    # Mock Figure.show to be a no-op
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage1_b(ctx):
    import plotly.graph_objects as go

    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage1_c(ctx):
    # New Stage 1c trains a tiny zero-layer model locally and renders with
    # plotly graph_objects; only the figure display needs mocking.
    import plotly.graph_objects as go

    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_a(ctx):
    # Stage 2a fetches a little Arabic text, builds a compact mGPT vocab, trains
    # a tiny 1-layer model, and renders attention heatmaps. Shrink the data and
    # training knobs so the notebook runs fast, and no-op the plotly display.
    import plotly.graph_objects as go

    ctx["MAX_CHARS"] = 15000
    ctx["N_CTX"] = 16
    ctx["N_EPOCHS"] = 2
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_b(ctx):
    # Stage 2b: like 2a but multi-head. Shrink data/training + few heads, no-op plotly.
    import plotly.graph_objects as go

    ctx["MAX_CHARS"] = 15000
    ctx["N_CTX"] = 16
    ctx["N_EPOCHS"] = 2
    ctx["N_HEADS"] = 4
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_c(ctx):
    # Stage 2c: synthetic induction training + a 2-layer Arabic model. Shrink both
    # (won't form a clean induction head at these sizes, but must run), no-op plotly.
    import plotly.graph_objects as go

    ctx["STEPS"] = 30          # synthetic induction training steps
    ctx["MAX_CHARS"] = 15000   # Arabic corpus
    ctx["N_CTX_AR"] = 16
    ctx["N_EPOCHS"] = 2
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_dash(ctx):
    # Stage 2dash: loads a faithful-scale trained checkpoint and decomposes it into
    # bigram + skip-trigram circuits. FORCE_TINY swaps in a tiny network-free model so
    # CI runs in seconds without the (gitignored) checkpoint or a HF download.
    import plotly.graph_objects as go

    ctx["FORCE_TINY"] = True
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_dash2(ctx):
    # Stage 2dash²: loads two trained checkpoints (1-layer 2dash + 2-layer) and
    # decomposes the two-layer model into composition circuits + an induction head.
    # FORCE_TINY swaps in tiny network-free models so CI runs in seconds without the
    # (gitignored) checkpoints or any HF download.
    import plotly.graph_objects as go

    ctx["FORCE_TINY"] = True
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


def mock_stage2_d(ctx):
    # Stage 2d: 2-layer model WITH MLP on Arabic. Shrink data/training, no-op plotly.
    import plotly.graph_objects as go

    ctx["MAX_CHARS"] = 15000
    ctx["N_CTX"] = 16
    ctx["N_EPOCHS"] = 2
    go.Figure.show = lambda self: print("  [Mock] plotly.Figure.show() called.")


# Run checks
# Resolve notebook paths relative to this file, not the caller's cwd, so the
# existence checks below never silently skip (a vacuous "SUCCESS") when the
# harness is invoked from the repo root.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
all_passed = True
stage_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

# Stage 1a
if stage_arg in ("a", "all"):
    if os.path.exists("stage1_a_char_reference.ipynb"):
        success = run_notebook("stage1_a_char_reference.ipynb", mock_stage1_a)
        if not success:
            all_passed = False

# Stage 1b
if stage_arg in ("b", "all"):
    if os.path.exists("stage1_b_word_reference.ipynb"):
        success = run_notebook("stage1_b_word_reference.ipynb", mock_stage1_b)
        if not success:
            all_passed = False

# Stage 1c
if stage_arg in ("c", "all"):
    if os.path.exists("stage1_c_subword_reference.ipynb"):
        success = run_notebook("stage1_c_subword_reference.ipynb", mock_stage1_c)
        if not success:
            all_passed = False

# Stage 2b
if stage_arg in ("2b", "all"):
    if os.path.exists("stage2_b_multi_head_reference.ipynb"):
        success = run_notebook("stage2_b_multi_head_reference.ipynb", mock_stage2_b)
        if not success:
            all_passed = False

# Stage 2c
if stage_arg in ("2c", "all"):
    if os.path.exists("stage2_c_depth_induction_reference.ipynb"):
        success = run_notebook("stage2_c_depth_induction_reference.ipynb", mock_stage2_c)
        if not success:
            all_passed = False

# Stage 2d
if stage_arg in ("2d", "all"):
    if os.path.exists("stage2_d_mlp_reference.ipynb"):
        success = run_notebook("stage2_d_mlp_reference.ipynb", mock_stage2_d)
        if not success:
            all_passed = False

# Stage 2dash
if stage_arg in ("2dash", "all"):
    if os.path.exists("stage2_dash_skip_trigram_reference.ipynb"):
        success = run_notebook("stage2_dash_skip_trigram_reference.ipynb", mock_stage2_dash)
        if not success:
            all_passed = False

# Stage 2dash²
if stage_arg in ("2dash2", "all"):
    if os.path.exists("stage2_dash2_composition_induction_reference.ipynb"):
        success = run_notebook("stage2_dash2_composition_induction_reference.ipynb", mock_stage2_dash2)
        if not success:
            all_passed = False

# Stage 2a
if stage_arg in ("2a", "all"):
    if os.path.exists("stage2_a_single_block_reference.ipynb"):
        success = run_notebook("stage2_a_single_block_reference.ipynb", mock_stage2_a)
        if not success:
            all_passed = False

if all_passed:
    print(f"\n🎉 REFERENCE NOTEBOOKS ({stage_arg}) VERIFIED SUCCESSFULLY!")
    sys.exit(0)
else:
    print(f"\n❌ SOME NOTEBOOKS ({stage_arg}) FAILED VERIFICATION.")
    sys.exit(1)
