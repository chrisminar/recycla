import tkinter as tk
from typing import Any, Callable, Dict, Optional

import numpy as np
from main import get_sub_materials, prep_model, update_cloud_gt, update_cloud_index
from PIL import Image, ImageTk

from recycla.connect.firebase_models import DataLabel

model, primary, secondary = prep_model()  # prep model to run


def get_current_key(index: int, data: Dict[str, Any]) -> str:
    """
    Return the key at the given index from the data dictionary.
    """
    return list(data.keys())[index]


def add_info_labels(
    window: tk.Tk,
    data: Dict[str, Any],
    current_index: int,
    stringvars: Optional[Dict[str, tk.StringVar]] = None,
) -> Dict[str, tk.StringVar]:
    """
    Add info labels (ID, path, material, sub-material, item number) to the window.
    Returns the StringVars used for material and sub-material.
    """
    current_key = get_current_key(current_index, data)
    info_frame = tk.Frame(window)
    info_frame.pack(pady=10)

    # Use provided StringVars or create new ones
    if stringvars is None:
        stringvars = {
            "gt_mat": tk.StringVar(value=data[current_key]["gt_mat"]),
            "gt_sub": tk.StringVar(value=data[current_key]["gt_sub"]),
        }

    labels = [
        ("ID", data[current_key]["id"]),
        ("h264 Path", data[current_key]["h264_path"].stem),
        ("Material", stringvars["gt_mat"]),
        ("Sub Material", stringvars["gt_sub"]),
        ("Item", f"{current_index+1}/{len(data)}"),
    ]

    for i, (label_text, value) in enumerate(labels):
        tk.Label(
            info_frame,
            text=f"{label_text}: ",
            anchor="w",
            width=10,
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=2 * i, sticky="w")
        if isinstance(value, tk.StringVar):
            tk.Label(
                info_frame, textvariable=value, anchor="w", width=30, font=("Arial", 10)
            ).grid(row=0, column=2 * i + 1, sticky="w")
        else:
            tk.Label(
                info_frame, text=value, anchor="w", width=30, font=("Arial", 10)
            ).grid(row=0, column=2 * i + 1, sticky="w")
    return stringvars


def add_scrollable_images(
    window: tk.Tk, data: Dict[str, Any], current_index: int
) -> None:
    """
    Add a scrollable frame of images with prediction and ground truth info, and valid/invalid buttons.
    """
    current_key = get_current_key(current_index, data)
    images = data[current_key]["images"]
    primary_probs = data[current_key]["primary_probabilities"]
    secondary_probs = data[current_key]["secondary_probabilities"]
    gt_match = data[current_key]["gt_match_index"]
    preview = data[current_key]["preview_index"]

    # Ensure valid_frames exists
    if "valid_frames" not in data[current_key]:
        data[current_key]["valid_frames"] = [
            0 if prob == "miscellaneous, background" else 1 for prob in secondary_probs
        ]

    container = tk.Frame(window)
    container.pack(side="left", fill="both", expand=True, anchor="n")
    canvas = tk.Canvas(container, height=400)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    image_refs = []

    def set_valid(idx, value, frame):
        data[current_key]["valid_frames"][idx] = value
        data[current_key]["most_recent_index_edited"] = idx
        print(f"Frame {idx} set to {'valid' if value == 1 else 'invalid'}")
        if value == 1:
            frame.configure(bg="#ccffcc")  # light green
        else:
            frame.configure(bg="#ffcccc")  # light red

    columns = 5
    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns

        frame = tk.Frame(scrollable_frame, bd=2, relief="groove", padx=5, pady=5)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nw")

        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # pil_img = pil_img.resize((128, 128))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_refs.append(tk_img)

        img_label = tk.Label(frame, image=tk_img)
        img_label.pack(side="top")

        info_text = (
            f"Frame: {idx}\n"
            f"Predicted Material: {primary_probs[idx]}\n"
            f"Predicted Submaterial: {secondary_probs[idx]}\n"
            f"GT Match: {gt_match[idx]}\n"
            f"Preview: {preview[idx]}"
        )
        info_label = tk.Label(
            frame,
            text=info_text,
            justify="left",
            font=("Arial", 10),
            width=36,  # Set fixed width for info label
            anchor="w",
        )
        info_label.pack(side="top", padx=10)

        # Buttons frame
        btn_frame = tk.Frame(frame)
        btn_frame.pack(side="top", pady=5)

        check_btn = tk.Button(
            btn_frame,
            text="✔️",
            fg="green",
            font=("Arial", 14, "bold"),
            width=3,
            command=lambda i=idx, f=frame: set_valid(i, 1, f),
        )
        check_btn.pack(side="left", padx=2)

        x_btn = tk.Button(
            btn_frame,
            text="❌",
            fg="red",
            font=("Arial", 14, "bold"),
            width=3,
            command=lambda i=idx, f=frame: set_valid(i, 0, f),
        )
        x_btn.pack(side="left", padx=2)

        # Set background color based on valid_frames
        if data[current_key]["valid_frames"][idx] == 0:
            frame.configure(bg="#ffcccc")  # light red
        if data[current_key]["valid_frames"][idx] == 1:
            frame.configure(bg="#ccffcc")

    scrollable_frame.image_refs = image_refs


def setup_mousewheel(window: tk.Tk) -> None:
    """
    Enable mousewheel scrolling for the image canvas in the window.
    """

    def _on_mousewheel(event: Any) -> None:
        # Find the canvas widget in the window's children
        for child in window.winfo_children():
            if isinstance(child, tk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, tk.Canvas):
                        # For Windows and MacOS
                        subchild.yview_scroll(int(-1 * (event.delta / 120)), "units")
                        return

    # Bind mousewheel for Windows and MacOS
    window.bind_all("<MouseWheel>", _on_mousewheel)
    # For Linux (event.num 4/5)
    window.bind_all(
        "<Button-4>",
        lambda event: _on_mousewheel(type("Event", (object,), {"delta": 120})()),
    )
    window.bind_all(
        "<Button-5>",
        lambda event: _on_mousewheel(type("Event", (object,), {"delta": -120})()),
    )


def add_gt_edit_frame(
    window: tk.Tk,
    data: Dict[str, Any],
    current_index: int,
    stringvars: Dict[str, tk.StringVar],
) -> tk.Frame:
    """
    Add dropdowns for editing ground truth material and sub-material for the current item.
    Returns the frame containing the controls.
    """
    current_key = get_current_key(current_index, data)
    # Create a frame to the right of the scrollable images
    gt_frame = tk.Frame(window, padx=20, pady=20)
    gt_frame.pack(side="right", fill="y", expand=False, anchor="n")

    # Dropdown for primary (material)
    tk.Label(gt_frame, text="Material:", font=("Arial", 12, "bold")).pack(anchor="w")
    primary_var = stringvars["gt_mat"]
    primary_menu = tk.OptionMenu(gt_frame, primary_var, *primary)
    primary_menu.config(width=20)
    primary_menu.pack(anchor="w", pady=(0, 10))

    # Dropdown for sub-material
    tk.Label(gt_frame, text="Sub Material:", font=("Arial", 12, "bold")).pack(
        anchor="w"
    )
    sub_var = stringvars["gt_sub"]
    sub_options = get_sub_materials(data[current_key]["gt_mat"], secondary)
    sub_menu = tk.OptionMenu(gt_frame, sub_var, *sub_options)
    sub_menu.config(width=20)
    sub_menu.pack(anchor="w", pady=(0, 10))

    def update_sub_options(*args):
        # Clear current menu
        menu = sub_menu["menu"]
        menu.delete(0, "end")
        # Get new options
        new_options = get_sub_materials(primary_var.get(), secondary)
        # Add new options
        for option in new_options:
            menu.add_command(
                label=option, command=lambda value=option: sub_var.set(value)
            )
        # Set to first option or empty if none
        if new_options:
            sub_var.set(new_options[0])
        else:
            sub_var.set("")

    # Callback to update gt_mat in data when dropdown changes
    def on_primary_change(*args):
        data[current_key]["gt_mat"] = primary_var.get()
        stringvars["gt_mat"].set(primary_var.get())
        update_sub_options()

    primary_var.trace_add("write", on_primary_change)

    # Callback to update gt_sub in data when dropdown changes
    def on_sub_change(*args):
        data[current_key]["gt_sub"] = sub_var.get()
        stringvars["gt_sub"].set(sub_var.get())
        model_gt = DataLabel(
            material=data[current_key]["gt_mat"],
            sub_material=data[current_key]["gt_sub"],
        )
        data[current_key]["model"].user_ground_truth = model_gt
        update_cloud_gt(data, current_key, model_gt)

    sub_var.trace_add("write", on_sub_change)

    return gt_frame


def add_valid_frames_edit_buttons(
    parent: tk.Frame,
    data: Dict[str, Any],
    current_key: str,
    refresh_callback: Optional[Callable[[], None]],
) -> tk.Frame:
    """
    Add buttons to set all frames before/after the most recently edited index to invalid.
    Returns the frame containing the buttons.
    """
    btn_frame = tk.Frame(parent, padx=20, pady=10)
    btn_frame.pack(side="top", fill="x", anchor="n")
    btn_container = tk.Frame(btn_frame)
    btn_container.pack()

    def set_all_before_invalid():
        idx = data[current_key].get("most_recent_index_edited")
        if idx is None:
            print("No index edited yet.")
            return
        for i in range(idx):
            data[current_key]["valid_frames"][i] = 0
        print(f"Set all frames before index {idx} to invalid.")
        if refresh_callback:
            refresh_callback()

    def set_all_after_invalid():
        idx = data[current_key].get("most_recent_index_edited")
        if idx is None:
            print("No index edited yet.")
            return
        print(f"Set all frames after index {idx} to invalid.")
        for i in range(idx + 1, len(data[current_key]["valid_frames"])):
            data[current_key]["valid_frames"][i] = 0
        if refresh_callback:
            refresh_callback()

    def reset_all_valid():
        """Set all frames to valid (1)."""
        data[current_key]["valid_frames"] = [1] * len(data[current_key]["valid_frames"])
        print("All frames set to valid.")
        if refresh_callback:
            refresh_callback()

    before_btn = tk.Button(
        btn_container, text="Set All Before To Invalid", command=set_all_before_invalid
    )
    before_btn.pack(side="left", padx=5)

    after_btn = tk.Button(
        btn_container, text="Set All After To Invalid", command=set_all_after_invalid
    )
    after_btn.pack(side="left", padx=5)

    reset_btn = tk.Button(
        btn_container, text="Reset All to Valid", command=reset_all_valid
    )
    reset_btn.pack(side="left", padx=5)
    return btn_frame


def add_next_frame(
    parent: tk.Frame,
    data: Dict[str, Any],
    current_index: int,
    rebuild_gui: Callable[[tk.Tk, Dict[str, Any], int], None],
    update_cloud: bool = True,
) -> None:
    """
    Add previous/next navigation buttons to the parent frame.
    """
    next_frame = tk.Frame(parent, padx=20, pady=10)
    next_frame.pack(side="top", fill="x", anchor="n")

    btn_container = tk.Frame(next_frame)
    btn_container.pack()

    def on_prev():
        if update_cloud:
            current_key = get_current_key(current_index, data)
            update_cloud_index(data, current_key)
        if current_index > 0:
            rebuild_gui(parent.master.master, data, current_index - 1)
        else:
            print("Already at first item.")

    def on_next():
        if update_cloud:
            current_key = get_current_key(current_index, data)
            update_cloud_index(data, current_key)
        if current_index + 1 < len(data):
            rebuild_gui(parent.master.master, data, current_index + 1)
        else:
            print("No more items.")

    def on_skip():
        current_key = get_current_key(current_index, data)
        # Set all frames to invalid
        data[current_key]["valid_frames"] = [0] * len(data[current_key]["valid_frames"])
        # Set ground truth to miscellaneous
        data[current_key]["gt_mat"] = "miscellaneous"
        data[current_key]["gt_sub"] = "miscellaneous"
        model_gt = DataLabel(material="miscellaneous", sub_material="miscellaneous")
        data[current_key]["model"].user_ground_truth = model_gt

        # Update cloud with new ground truth and valid frames
        if update_cloud:
            update_cloud_gt(data, current_key, model_gt)
            update_cloud_index(data, current_key)

        print(
            f"Skipped item {current_index + 1}: set all frames invalid and changed to miscellaneous"
        )

        # Go to next item
        if current_index + 1 < len(data):
            rebuild_gui(parent.master.master, data, current_index + 1)
        else:
            print("No more items.")

    prev_btn = tk.Button(
        btn_container, text="◀ Previous", font=("Arial", 20, "bold"), command=on_prev
    )
    prev_btn.pack(side="left", padx=(0, 20))

    skip_btn = tk.Button(
        btn_container,
        text="Skip ⏭",
        font=("Arial", 20, "bold"),
        command=on_skip,
        fg="orange",
    )
    skip_btn.pack(side="left", padx=(0, 20))

    next_btn = tk.Button(
        btn_container, text="Next ▶", font=("Arial", 20, "bold"), command=on_next
    )
    next_btn.pack(side="left")


def add_preview_images(
    parent: tk.Frame,
    data: Dict[str, Any],
    current_index: int,
) -> None:
    """
    Adds a frame with preview images to the parent widget.
    """
    current_key = get_current_key(current_index, data)
    images = data[current_key]["images"]
    preview_flags = data[current_key].get("preview_index", [])

    preview_images_data = []
    for idx, is_preview in enumerate(preview_flags):
        if is_preview:
            if len(preview_images_data) < 3:
                preview_images_data.append((idx, images[idx]))

    if not preview_images_data:
        return

    preview_frame = tk.Frame(parent, padx=20, pady=10)
    preview_frame.pack(side="top", fill="x", anchor="n")

    tk.Label(preview_frame, text="Preview Images", font=("Arial", 12, "bold")).pack(
        anchor="w"
    )

    images_container = tk.Frame(preview_frame)
    images_container.pack()

    image_refs = []
    for idx, img_data in preview_images_data:
        img_frame = tk.Frame(images_container, bd=1, relief="solid", padx=5, pady=5)
        img_frame.pack(side="left", padx=5, pady=5)

        pil_img = Image.fromarray(np.transpose(img_data, (1, 2, 0)))
        pil_img = pil_img.resize((128, 128))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_refs.append(tk_img)

        img_label = tk.Label(img_frame, image=tk_img)
        img_label.pack(side="top")

        frame_num_label = tk.Label(img_frame, text=f"Frame: {idx}")
        frame_num_label.pack(side="top")

    preview_frame.image_refs = image_refs


def build_gui(window: tk.Tk, data: Dict[str, Any], current_index: int) -> None:
    """
    Build or rebuild the main GUI for the labeling tool at the given index.
    """
    # reset window
    for widget in window.winfo_children():
        widget.destroy()

    current_key = get_current_key(current_index, data)
    stringvars = {
        "gt_mat": tk.StringVar(value=data[current_key]["gt_mat"]),
        "gt_sub": tk.StringVar(value=data[current_key]["gt_sub"]),
    }
    add_info_labels(window, data, current_index, stringvars)
    add_scrollable_images(window, data, current_index)
    setup_mousewheel(window)
    gt_frame = add_gt_edit_frame(window, data, current_index, stringvars)
    quick_switch_frame = add_valid_frames_edit_buttons(
        gt_frame, data, current_key, lambda: build_gui(window, data, current_index)
    )
    add_next_frame(quick_switch_frame, data, current_index, build_gui)
    add_preview_images(gt_frame, data, current_index)
