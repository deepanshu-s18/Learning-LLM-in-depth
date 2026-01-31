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
