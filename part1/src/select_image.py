"""
Select an image using a file dialog and return the selected path.

Provides function `select_image()` which returns the selected image path as a string.
When run as CLI it prints the path to stdout.

If the Tk dialog cannot be opened (headless), falls back to a reasonable default
`sample_input.png` located in the SecondHalf workspace.
"""
import os
import sys


def select_image() -> str:
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        root = Tk()
        root.withdraw()
        filetypes = [("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")]
        path = askopenfilename(title="Select Aadhaar / Non-Aadhaar Image", filetypes=filetypes)
        root.update()
        root.destroy()

        if path:
            return path

    except Exception:
        # GUI not available or user cancelled
        pass

    # Fallback path: sample_input.png located in SecondHalf workspace
    fallback = os.path.join(os.path.expanduser(r"~"), "OneDrive", "Desktop", "ML Project", "SecondHalf", "sample_input.png")
    if os.path.exists(fallback):
        return fallback

    # Last resort: current working dir file
    cwd_fallback = os.path.join(os.getcwd(), "sample_input.png")
    return cwd_fallback


if __name__ == '__main__':
    path = select_image()
    print(path)
