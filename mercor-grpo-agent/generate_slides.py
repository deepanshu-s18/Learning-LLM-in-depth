import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_paradigm_slide(output_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200, facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    c_bg = '#181818'
    c_border_neutral = '#4a4a4a'
    c_teacher = '#22c55e'   # Green
    c_student = '#0ea5e9'   # Cyan/Blue
    c_env = '#ef4444'       # Coral/Red
    c_text_main = '#f3f4f6'
    c_text_sub = '#9ca3af'

    columns = [
        {"name": "SFT", "x": 4.5, 
         "w_title": "TEACHER writes", "w_sub": "the solution", "w_col": c_teacher,
         "g_title": "STUDENT learns it", "g_sub": "token by token", "g_col": c_student},
        {"name": "RL", "x": 8.5, 
         "w_title": "STUDENT writes", "w_sub": "its own attempt", "w_col": c_student,
         "g_title": "ENVIRONMENT grades", "g_sub": "one number", "g_col": c_env},
        {"name": "OPD", "x": 12.5, 
         "w_title": "STUDENT writes", "w_sub": "its own attempt", "w_col": c_student,
         "g_title": "TEACHER grades", "g_sub": "every token", "g_col": c_teacher},
    ]

    # Row labels on left
    ax.text(1.8, 5.0, "who writes", color='#888888', fontsize=18, fontweight='500', va='center', ha='right', family='sans-serif')
    ax.text(1.8, 3.0, "who grades", color='#888888', fontsize=18, fontweight='500', va='center', ha='right', family='sans-serif')

    box_w = 2.9
    box_h = 0.95

    def draw_box(x, y, title, subtitle=None, border_color='#ffffff', title_color='#ffffff'):
        rect = patches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=2.2, edgecolor=border_color, facecolor=c_bg, zorder=3
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x, y + 0.16, title, color=title_color, fontsize=15, fontweight='700', ha='center', va='center', family='sans-serif', zorder=4)
            ax.text(x, y - 0.20, subtitle, color=c_text_sub, fontsize=12, fontweight='400', ha='center', va='center', family='sans-serif', zorder=4)
        else:
            ax.text(x, y, title, color=title_color, fontsize=16, fontweight='600', ha='center', va='center', family='sans-serif', zorder=4)

    def draw_arrow(x, y_start, y_end):
        ax.annotate(
            '', xy=(x, y_end), xytext=(x, y_start),
            arrowprops=dict(arrowstyle="-|>", color='#666666', lw=2.0, mutation_scale=16),
            zorder=2
        )

    for col in columns:
        x = col["x"]
        # Column title
        ax.text(x, 8.0, col["name"], color='#ffffff', fontsize=26, fontweight='800', ha='center', va='center', family='sans-serif')

        # Row 1: prompt
        draw_box(x, 6.8, "prompt", border_color=c_border_neutral, title_color=c_text_main)
        draw_arrow(x, 6.25, 5.55)

        # Row 2: who writes
        draw_box(x, 5.0, col["w_title"], col["w_sub"], border_color=col["w_col"], title_color=col["w_col"])
        draw_arrow(x, 4.45, 3.55)

        # Row 3: who grades
        draw_box(x, 3.0, col["g_title"], col["g_sub"], border_color=col["g_col"], title_color=col["g_col"])
        draw_arrow(x, 2.45, 1.75)

        # Row 4: update
        draw_box(x, 1.2, "update", border_color=c_border_neutral, title_color=c_text_main)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='#121212', bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

create_paradigm_slide('/Users/shobhitagnihotri/Desktop/internship/mercor-grpo-agent/docs/paradigm_comparison_slide.png')
