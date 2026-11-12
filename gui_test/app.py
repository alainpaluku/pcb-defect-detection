"""EN: Launch the optional Tk desktop app. FR: Lancer l'application Tk facultative."""

from .utils import setup_logging, validate_dependencies


def main():
    setup_logging()
    missing = validate_dependencies()
    if missing:
        raise SystemExit("Missing dependencies / Dépendances manquantes: " + ", ".join(missing))

    import ttkbootstrap as ttk

    from .main_window import PCBDetectionGUI

    root = ttk.Window(themename="cosmo")
    PCBDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
