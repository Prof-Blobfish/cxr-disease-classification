# Label equipment and chest drains on expert-labelled NIH test images, to measure two pneumothorax
# shortcuts: "this patient has medical equipment" and "this patient has a chest drain".
#   python src/label_drains.py --pass equipment   # all images: any equipment vs none
#   python src/label_drains.py --pass drain       # images with equipment: chest drain vs other only
#
# Images: every four-findings test image with an expert-confirmed pneumothorax plus a random sample
# of expert-negative images, shuffled, without saying which is which. Labels are saved after every
# key press to annotations/<pass>_labels.csv; rerunning resumes.
# Keys: the pass's label keys, u = unsure, left arrow / backspace = previous image,
#       c = toggle contrast enhancement, q = save and quit.

from __future__ import annotations

import argparse
import tkinter as tk
from datetime import datetime, timezone

import pandas as pd
from PIL import Image, ImageOps, ImageTk

from data import load_bundle
from expert_eval import PREFIX, labels_dir, yes
from utils import PROJECT_ROOT

ANNOTATIONS = PROJECT_ROOT / "annotations"
NEGATIVES = 200
PASSES = {
    "equipment": {
        "keys": {"e": "any_equipment", "n": "no_equipment", "u": "unsure"},
        "help": "[e] any tube, line, wire or device   [n] none",
    },
    "drain": {
        "keys": {"d": "drain", "o": "other_equipment", "u": "unsure"},
        "help": "[d] CHEST DRAIN: tube entering through the side of the chest wall between the ribs   "
                "[o] other equipment only: ECG leads, breathing/feeding tube, central line, pacemaker",
    },
}


def worklist(pass_name: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_dir() / f"{PREFIX}four_findings_expert_labels_test_labels.csv")
    expert = yes(labels["Pneumothorax"])
    positives = labels.loc[expert == 1, "Image Index"]
    negatives = labels.loc[expert == 0, "Image Index"].sample(NEGATIVES, random_state=0)
    ids = pd.concat([positives, negatives]).sample(frac=1, random_state=0).tolist()
    if pass_name == "drain":  # only images where the equipment pass found something
        equipment = pd.read_csv(ANNOTATIONS / "equipment_labels.csv").set_index("image_id")["equipment"]
        ids = [image_id for image_id in ids if equipment.get(image_id) == "any_equipment"]
    paths = load_bundle({"split_seed": 0}).test_df.set_index("image_id")["image_path"]
    return pd.DataFrame({"image_id": ids, "image_path": paths.loc[ids].to_numpy()})


class Labeler:
    def __init__(self, items: pd.DataFrame, pass_name: str) -> None:
        self.items, self.spec = items, PASSES[pass_name]
        self.out, self.column = ANNOTATIONS / f"{pass_name}_labels.csv", pass_name
        self.labels = pd.read_csv(self.out).set_index("image_id")[self.column].to_dict() if self.out.is_file() else {}
        self.index = next((i for i, image in enumerate(items.image_id) if image not in self.labels), len(items) - 1)
        self.enhance = True
        self.root = tk.Tk()
        self.root.title(f"Labelling: {pass_name}")
        self.status = tk.Label(self.root, font=("Segoe UI", 12), justify="left", wraplength=1000)
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
        side = min(self.root.winfo_screenheight() - 170, 1000)
        image.thumbnail((side, side))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=self.photo)
        done = sum(image_id in self.labels for image_id in self.items.image_id)
        self.status.configure(text=f"{self.index + 1}/{len(self.items)}   labelled {done}   this image: "
                                   f"{self.labels.get(row.image_id, '-')}\n{self.spec['help']}   [u] unsure   "
                                   f"[←] back   [c] contrast {'on' if self.enhance else 'off'}   [q] quit")

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": list(self.labels), self.column: list(self.labels.values()),
                      "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}).to_csv(self.out, index=False)

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in self.spec["keys"]:
            self.labels[self.items.image_id.iloc[self.index]] = self.spec["keys"][key]
            self.save()
            if all(image_id in self.labels for image_id in self.items.image_id):
                self.status.configure(text=f"All {len(self.items)} images labelled. Saved to {self.out}. Press q to close.")
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
    parser = argparse.ArgumentParser(description="Label equipment / chest drains for the pneumothorax shortcut check.")
    parser.add_argument("--pass", dest="pass_name", choices=list(PASSES), required=True)
    args = parser.parse_args()
    items = worklist(args.pass_name)
    print(f"{len(items)} images. Labels: {ANNOTATIONS / f'{args.pass_name}_labels.csv'}")
    Labeler(items, args.pass_name).root.mainloop()


if __name__ == "__main__":
    main()
