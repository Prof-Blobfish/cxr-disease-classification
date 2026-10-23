# Label chest drains on expert-labelled NIH test images, to measure the pneumothorax drain shortcut:
#   python src/label_drains.py
#
# Shows every four-findings test image with an expert-confirmed pneumothorax plus a random sample
# of expert-negative images, shuffled, without saying which is which. Keys:
#   d = chest drain present    n = no chest drain    u = unsure
#   left arrow / backspace = previous image    c = toggle contrast enhancement    q = save and quit
# A chest drain is a tube entering the pleural space through the chest wall (often a pigtail
# catheter, usually lateral). Endotracheal tubes, feeding tubes and central lines do not count.
# Labels are saved after every key press to annotations/drain_labels.csv; rerunning resumes.

from __future__ import annotations

import tkinter as tk
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageTk

from data import load_bundle
from expert_eval import PREFIX, labels_dir, yes
from utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "annotations" / "drain_labels.csv"
NEGATIVES = 200
KEYS = {"d": "drain", "n": "no_drain", "u": "unsure"}


def worklist() -> pd.DataFrame:
    labels = pd.read_csv(labels_dir() / f"{PREFIX}four_findings_expert_labels_test_labels.csv")
    expert = yes(labels["Pneumothorax"])
    positives = labels.loc[expert == 1, "Image Index"]
    negatives = labels.loc[expert == 0, "Image Index"].sample(NEGATIVES, random_state=0)
    ids = pd.concat([positives, negatives]).sample(frac=1, random_state=0).tolist()
    paths = load_bundle({"split_seed": 0}).test_df.set_index("image_id")["image_path"]
    return pd.DataFrame({"image_id": ids, "image_path": paths.loc[ids].to_numpy()})


class Labeler:
    def __init__(self, items: pd.DataFrame) -> None:
        self.items = items
        self.labels = pd.read_csv(OUT).set_index("image_id")["drain"].to_dict() if OUT.is_file() else {}
        self.index = next((i for i, image in enumerate(items.image_id) if image not in self.labels), len(items) - 1)
        self.enhance = True
        self.root = tk.Tk()
        self.root.title("Chest drain labelling")
        self.status = tk.Label(self.root, font=("Segoe UI", 12), justify="left")
        self.status.pack(anchor="w", padx=8, pady=4)
        self.canvas = tk.Label(self.root, bg="black")
        self.canvas.pack()
        self.root.bind("<Key>", self.on_key)
        self.show()

    def show(self) -> None:
        row = self.items.iloc[self.index]
        image = Image.open(row.image_path).convert("L")
        if self.enhance:
            image = ImageOps.equalize(image)
        side = min(self.root.winfo_screenheight() - 140, 1000)
        image.thumbnail((side, side))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=self.photo)
        done = sum(image_id in self.labels for image_id in self.items.image_id)
        current = self.labels.get(row.image_id, "-")
        self.status.configure(text=f"{self.index + 1}/{len(self.items)}   labelled {done}   this image: {current}   "
                                   f"[d] drain  [n] no drain  [u] unsure  [←] back  [c] contrast {'on' if self.enhance else 'off'}  [q] quit")

    def save(self) -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": list(self.labels), "drain": list(self.labels.values()),
                      "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}).to_csv(OUT, index=False)

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in KEYS:
            self.labels[self.items.image_id.iloc[self.index]] = KEYS[key]
            self.save()
            if self.index == len(self.items) - 1 and all(i in self.labels for i in self.items.image_id):
                self.status.configure(text=f"All {len(self.items)} images labelled. Saved to {OUT}. Press q to close.")
                return
            self.index = min(self.index + 1, len(self.items) - 1)
        elif key in ("left", "backspace"):
            self.index = max(self.index - 1, 0)
        elif key == "c":
            self.enhance = not self.enhance
        elif key == "q":
            self.save()
            self.root.destroy()
            return
        else:
            return
        self.show()


def main() -> None:
    items = worklist()
    print(f"{len(items)} images ({NEGATIVES} without pneumothorax, shuffled). Labels: {OUT}")
    Labeler(items).root.mainloop()


if __name__ == "__main__":
    main()
