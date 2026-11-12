"""EN: Marshal worker results onto Tk's main thread. FR: Rapatrier les résultats sur le fil Tk."""

from queue import Empty, Queue
from threading import Thread
from tkinter import TclError


def run_background(parent, work, on_success, on_error, on_progress=None):
    """EN: Only poll() touches Tk widgets. FR: Seul poll() accède aux widgets Tk."""
    events = Queue()

    def worker():
        try:
            result = work(lambda value: events.put(("progress", value)))
            events.put(("success", result))
        except Exception as error:
            events.put(("error", error))

    def poll():
        try:
            if not parent.winfo_exists():
                return
            while True:
                try:
                    kind, value = events.get_nowait()
                except Empty:
                    break
                if kind == "success":
                    on_success(value)
                    return
                if kind == "error":
                    on_error(value)
                    return
                if on_progress is not None:
                    on_progress(value)
            parent.after(50, poll)
        except TclError:
            # EN: Closing a window cancels delivery. FR: Fermer la fenêtre annule la livraison.
            return

    Thread(target=worker, daemon=True).start()
    parent.after(50, poll)
