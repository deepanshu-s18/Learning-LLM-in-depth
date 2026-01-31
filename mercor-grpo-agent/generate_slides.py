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
