import pandas as pd
import plotly.express as px

def main():
    # 1. Load collected data
    try:
        df = pd.read_csv("attention_divergence_summary.csv")
    except Exception as e:
        print(f"Error loading summary CSV: {e}")
        return

    grouped = df.groupby(["layer", "head"]).mean(numeric_only=True).reset_index()

    # 2. Pivot
    entropy_pivot = grouped.pivot(index="layer", columns="head", values="entropy_diff")

    # 3. Plot with Masri labels
    fig = px.imshow(
        entropy_pivot.values,
        labels=dict(
            x="رأس الانتباه (Attention Head)", 
            y="الطبقة (Layer)", 
            color="فرق إنتروبيا الانتباه (مصري - فصحى)"
        ),
        x=entropy_pivot.columns,
        y=entropy_pivot.index,
        color_continuous_scale="RdBu",
        title="تباعد إنتروبيا الانتباه (عامية مصرية ضد فصحى) عبر طبقات Pythia-160m"
    )
    
    # Center title and set clean layout
    fig.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12)
    )

    # 4. Save
    fig.write_html("divergence_heatmap_masri.html")
    print("Saved Masri heatmap to divergence_heatmap_masri.html successfully!")

if __name__ == "__main__":
    main()
