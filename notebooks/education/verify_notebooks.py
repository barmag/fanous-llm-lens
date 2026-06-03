import json
import sys
import os

def run_notebook(filepath, mock_setup=None):
    print(f"\n==========================================")
    print(f"Testing Notebook: {os.path.basename(filepath)}")
    print(f"==========================================")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Extract code cells
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    
    # Build complete python source code
    source_lines = []
    for cell in code_cells:
        # Join lines of the cell
        cell_source = "".join(cell['source'])
        # Comment out colab specific pip installs to avoid running them locally
        # and insert 'pass' to maintain valid python indentation block
        cell_source = cell_source.replace("!pip install", "pass # !pip install")
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
    except Exception as e:
        import traceback
        print("Result: FAILED", file=sys.stderr)
        traceback.print_exc()
        return False

# Mock setups to make tests run instantly
def mock_stage1_a(ctx):
    # Mock clean_and_collect to only fetch 2000 chars instead of 100000
    # Also mock epochs to 1 for super fast training
    ctx['sys'] = sys
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

# Run checks
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

if all_passed:
    print(f"\n🎉 REFERENCE NOTEBOOKS ({stage_arg}) VERIFIED SUCCESSFULLY!")
    sys.exit(0)
else:
    print(f"\n❌ SOME NOTEBOOKS ({stage_arg}) FAILED VERIFICATION.")
    sys.exit(1)
