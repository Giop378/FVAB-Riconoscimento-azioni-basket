from pathlib import Path
import py_compile

path = Path("src/long_video/infer_exp46_from_store.py")

if not path.exists():
    raise FileNotFoundError(f"File non trovato: {path}")

s = path.read_text(encoding="utf-8")

start = s.index("def output_fieldnames()")
end = s.index("def write_prediction_rows", start)
block = s[start:end]

if '"dino_num_frames"' not in block:
    old = '        "num_store_samples",\n'
    new = (
        '        "num_store_samples",\n'
        '        "dino_num_frames",\n'
        '        "tracking_raw_num_frames",\n'
        '        "model_input_length",\n'
    )

    if old not in block:
        raise RuntimeError("Non trovo la riga num_store_samples dentro output_fieldnames().")

    block = block.replace(old, new, 1)
    s = s[:start] + block + s[end:]

path.write_text(s, encoding="utf-8")
py_compile.compile(str(path), doraise=True)

print("[OK] Patch applicata e file compilato correttamente.")
