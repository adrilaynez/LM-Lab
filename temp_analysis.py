import json
import os

paths = [
    r"c:\Projects\LM-Lab\checkpoints\pedagogical\depth_comparison\summary.json",
    r"c:\Projects\LM-Lab\checkpoints\pedagogical\depth_high_lr\summary.json"
]
with open("c:\\Projects\\LM-Lab\\temp_analysis_output_utf8.txt", "w", encoding="utf-8") as out:
    for p in paths:
        out.write(f"\n--- {os.path.basename(os.path.dirname(p))} ---\n")
        data = json.load(open(p, "r"))
        for r in data:
            layer = r["config"]["num_layers"]
            lr = r["config"]["learning_rate"]
            prms = r["total_params"]
            tr_loss = r["final_train_loss"]
            vl_loss = r["final_val_loss"]
            div = r.get("diverged", False)
            out.write(f"Model: {r['label']:15} | L: {layer:2} | LR: {lr:5} | Params: {prms:6} | Train L: {tr_loss:.4f} | Val L: {vl_loss:.4f} | Diverged: {div}\n")
