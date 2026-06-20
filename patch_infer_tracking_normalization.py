from pathlib import Path
import re
import py_compile

path = Path("src/long_video/infer_exp46_from_store.py")
if not path.exists():
    path = Path("infer_exp46_from_store.py")

if not path.exists():
    raise FileNotFoundError("Non trovo infer_exp46_from_store.py. Lancia questo script dalla root della repository.")

s = path.read_text(encoding="utf-8")

# =============================================================================
# 1. Aggiorna nome policy
# =============================================================================

s = s.replace(
    'TEMPORAL_POLICY = "train_like_dino_all_frames_tracking_max_48"',
    'TEMPORAL_POLICY = "train_like_dino_all_frames_tracking_max_48_tracking_normalized"',
)

# =============================================================================
# 2. Estende LevelBundle con normalizzatore tracking
# =============================================================================

if "tracking_normalized: bool" not in s:
    s = s.replace(
        "    labels: list[str]\n"
        "    model: nn.Module\n",
        "    labels: list[str]\n"
        "    model: nn.Module\n"
        "    tracking_normalized: bool\n"
        "    tracking_mean: np.ndarray\n"
        "    tracking_std: np.ndarray\n",
        1,
    )

# =============================================================================
# 3. Aggiunge funzioni per leggere/applicare normalizzazione checkpoint
# =============================================================================

normalizer_functions = r'''

def get_tracking_normalizer(
    name: str,
    tracking_config: dict[str, Any],
    num_features: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Legge mean/std dal checkpoint per replicare la normalizzazione del training."""
    normalized = bool(tracking_config.get("normalized", False))
    mean_value = tracking_config.get("mean")
    std_value = tracking_config.get("std")

    if normalized:
        if mean_value is None or std_value is None:
            raise ValueError(
                f"{name}: tracking_config.normalized=True ma mean/std sono assenti nel checkpoint."
            )
        mean = np.asarray(mean_value, dtype=np.float32)
        std = np.asarray(std_value, dtype=np.float32)
    else:
        mean = np.zeros((num_features,), dtype=np.float32)
        std = np.ones((num_features,), dtype=np.float32)

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            f"{name}: mean/std tracking devono essere vettori 1D, "
            f"trovati {mean.shape} e {std.shape}."
        )

    if mean.shape[0] != int(num_features) or std.shape[0] != int(num_features):
        raise ValueError(
            f"{name}: dimensione mean/std tracking non coerente. "
            f"Feature={num_features}, mean={mean.shape[0]}, std={std.shape[0]}."
        )

    std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def apply_tracking_normalization(tracking_seq: np.ndarray, level: LevelBundle) -> np.ndarray:
    """Applica z-score alle feature tracking [T, K] se il checkpoint lo richiede."""
    tracking_seq = np.asarray(tracking_seq, dtype=np.float32)

    if tracking_seq.ndim != 2:
        raise ValueError(
            f"{level.name}: tracking_seq deve avere shape [T, K], trovato {tracking_seq.shape}."
        )

    if tracking_seq.shape[1] != len(level.feature_names):
        raise ValueError(
            f"{level.name}: tracking_seq ha K={tracking_seq.shape[1]}, "
            f"ma il checkpoint richiede K={len(level.feature_names)}."
        )

    if not level.tracking_normalized:
        return tracking_seq.astype(np.float32)

    return (
        (tracking_seq - level.tracking_mean.reshape(1, -1))
        / level.tracking_std.reshape(1, -1)
    ).astype(np.float32)
'''

if "def get_tracking_normalizer(" not in s:
    s = s.replace(
        "\ndef load_level_bundle(\n",
        normalizer_functions + "\n\ndef load_level_bundle(\n",
        1,
    )

# =============================================================================
# 4. Dentro load_level_bundle legge mean/std dal tracking_config
# =============================================================================

if "tracking_normalized, tracking_mean, tracking_std = get_tracking_normalizer" not in s:
    marker = "    if state_dict is not None:\n"
    insert = (
        "    tracking_normalized, tracking_mean, tracking_std = get_tracking_normalizer(\n"
        "        name=name,\n"
        "        tracking_config=tracking_config,\n"
        "        num_features=len(feature_names),\n"
        "    )\n\n"
    )
    s = s.replace(marker, insert + marker, 1)

if 'print(f"     tracking normalized: {tracking_normalized}")' not in s:
    s = s.replace(
        '    print(f"     tracking features: {len(feature_names)}")\n',
        '    print(f"     tracking features: {len(feature_names)}")\n'
        '    print(f"     tracking normalized: {tracking_normalized}")\n',
        1,
    )

if "tracking_mean=tracking_mean" not in s:
    s = s.replace(
        "        labels=labels,\n"
        "        model=model,\n",
        "        labels=labels,\n"
        "        model=model,\n"
        "        tracking_normalized=tracking_normalized,\n"
        "        tracking_mean=tracking_mean,\n"
        "        tracking_std=tracking_std,\n",
        1,
    )

# =============================================================================
# 5. Sostituisce build_input_for_window con versione normalizzata
# =============================================================================

new_build_input_for_window = r'''def build_input_for_window(
    feature_store: FeatureStore,
    row: WindowRow,
    level: LevelBundle,
    primitives: dict[str, np.ndarray],
) -> np.ndarray:
    # Train-like:
    # - DINO usa tutti i sample/frame reali della finestra;
    # - tracking viene calcolato su max 48 frame uniformi;
    # - tracking viene interpolato a T DINO;
    # - tracking viene normalizzato con mean/std del checkpoint;
    # - infine DINO e tracking vengono concatenati per timestep.
    start_idx = int(row.store_start_index)
    end_idx = int(row.store_end_index)

    if start_idx < 0 or end_idx > feature_store.timestamps.shape[0] or end_idx <= start_idx:
        raise ValueError(
            f"Indici finestra non validi per {row.window_id}: "
            f"{start_idx}:{end_idx} su N={feature_store.timestamps.shape[0]}"
        )

    dino_seq = np.asarray(feature_store.dino_features[start_idx:end_idx], dtype=np.float32)
    dino_times = feature_store.timestamps[start_idx:end_idx].astype(np.float64)
    dino_len = int(dino_seq.shape[0])

    if dino_len <= 0:
        raise ValueError(f"Finestra senza feature DINO: {row.window_id}")

    tracking_local_indices = select_uniform_local_indices(
        num_items=dino_len,
        max_items=TRACKING_MAX_FRAMES_PER_WINDOW,
    )

    if tracking_local_indices.size == 0:
        raise ValueError(f"Finestra senza frame tracking: {row.window_id}")

    tracking_query_times = dino_times[tracking_local_indices]

    tracking_raw_seq = build_tracking_sequence(
        store_timestamps=feature_store.timestamps,
        primitives=primitives,
        feature_names=level.feature_names,
        query_times=tracking_query_times,
        start_time=row.start_time,
        end_time=row.end_time,
        velocity_mode=TRACKING_VELOCITY_MODE,
    )

    tracking_seq = interpolate_sequence_array(tracking_raw_seq, target_len=dino_len)
    tracking_seq = apply_tracking_normalization(tracking_seq, level=level)

    if dino_seq.shape[0] != tracking_seq.shape[0]:
        raise RuntimeError(
            f"T diverso tra DINO e tracking: {dino_seq.shape} vs {tracking_seq.shape}"
        )

    return np.concatenate([dino_seq, tracking_seq], axis=1).astype(np.float32)
'''

pattern = r"def build_input_for_window\(\n.*?(?=\ndef pad_sequences\()"
s_new = re.sub(pattern, new_build_input_for_window + "\n\n", s, count=1, flags=re.S)

if s_new == s:
    raise RuntimeError("Non sono riuscito a sostituire build_input_for_window. Controlla il file.")
s = s_new

# =============================================================================
# 6. Aggiorna le chiamate a build_input_for_window in build_batch_inputs
# =============================================================================

s = s.replace("feature_names=l1.feature_names,\n", "level=l1,\n")
s = s.replace("feature_names=l2.feature_names,\n", "level=l2,\n")
s = s.replace("feature_names=l3.feature_names,\n", "level=l3,\n")

# =============================================================================
# 7. Aggiunge tracking_normalized nel summary inferenza
# =============================================================================

if '"tracking_normalized": {' not in s:
    s = s.replace(
        '        "tracking_max_frames_per_window": TRACKING_MAX_FRAMES_PER_WINDOW,\n',
        '        "tracking_max_frames_per_window": TRACKING_MAX_FRAMES_PER_WINDOW,\n'
        '        "tracking_normalized": {\n'
        '            "L1": bool(l1.tracking_normalized),\n'
        '            "L2": bool(l2.tracking_normalized),\n'
        '            "L3": bool(l3.tracking_normalized),\n'
        '        },\n',
        1,
    )

# =============================================================================
# 8. Aggiorna metadata runtime
# =============================================================================

s = s.replace(
    '"tracking_policy": "uniform_max_48_then_interpolate_to_dino_length",',
    '"tracking_policy": "uniform_max_48_then_interpolate_to_dino_length_then_checkpoint_zscore",',
)

level_replacements = {
    '"num_tracking_features": len(l1.feature_names),\n                "labels": l1.labels,':
        '"num_tracking_features": len(l1.feature_names),\n'
        '                "tracking_normalized": bool(l1.tracking_normalized),\n'
        '                "tracking_mean_len": int(l1.tracking_mean.shape[0]),\n'
        '                "tracking_std_len": int(l1.tracking_std.shape[0]),\n'
        '                "labels": l1.labels,',
    '"num_tracking_features": len(l2.feature_names),\n                "labels": l2.labels,':
        '"num_tracking_features": len(l2.feature_names),\n'
        '                "tracking_normalized": bool(l2.tracking_normalized),\n'
        '                "tracking_mean_len": int(l2.tracking_mean.shape[0]),\n'
        '                "tracking_std_len": int(l2.tracking_std.shape[0]),\n'
        '                "labels": l2.labels,',
    '"num_tracking_features": len(l3.feature_names),\n                "labels": l3.labels,':
        '"num_tracking_features": len(l3.feature_names),\n'
        '                "tracking_normalized": bool(l3.tracking_normalized),\n'
        '                "tracking_mean_len": int(l3.tracking_mean.shape[0]),\n'
        '                "tracking_std_len": int(l3.tracking_std.shape[0]),\n'
        '                "labels": l3.labels,',
}

for old, new in level_replacements.items():
    if old in s:
        s = s.replace(old, new, 1)

# =============================================================================
# 9. Stampa a runtime se i tre livelli stanno normalizzando
# =============================================================================

if 'print("tracking normalized:")' not in s:
    s = s.replace(
        '    print(f"velocity_mode:      {TRACKING_VELOCITY_MODE}")\n',
        '    print(f"velocity_mode:      {TRACKING_VELOCITY_MODE}")\n'
        '    print("tracking normalized:")\n'
        '    print(f"  L1: {l1.tracking_normalized}")\n'
        '    print(f"  L2: {l2.tracking_normalized}")\n'
        '    print(f"  L3: {l3.tracking_normalized}")\n',
        1,
    )

# =============================================================================
# 10. Salva e verifica compilazione
# =============================================================================

path.write_text(s, encoding="utf-8")
py_compile.compile(str(path), doraise=True)

print(f"[OK] File aggiornato e compilato correttamente: {path}")
print()
print("Controlli rapidi:")
print("- get_tracking_normalizer:", "get_tracking_normalizer" in s)
print("- apply_tracking_normalization:", "apply_tracking_normalization" in s)
print("- tracking_normalized:", "tracking_normalized" in s)
print("- policy normalized:", "train_like_dino_all_frames_tracking_max_48_tracking_normalized" in s)
