"""Shared paths and restrained visual style for submission figures."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.container import BarContainer, ErrorbarContainer  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.quiver import Quiver  # noqa: E402
from matplotlib.spines import Spine  # noqa: E402
from matplotlib.text import Text  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("GWT_SOURCE_DATA", PROJECT / "manuscript/source_data"))
OUTPUT = Path(os.environ.get("GWT_FIGURE_OUTPUT", PROJECT / "manuscript/figures"))
OUTPUT.mkdir(parents=True, exist_ok=True)
ARTIST_MANIFEST_DIR = Path(
    os.environ.get(
        "GWT_ARTIST_MANIFEST_DIR",
        PROJECT / "manuscript" / "source_data" / "figure_artist_manifests",
    )
)
ARTIST_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DOUBLE_COLUMN_MM = 180.0
MIN_FINAL_FONT_PT = 6.4
MAX_FINAL_FONT_PT = 9.8
MIN_FINAL_STROKE_PT = 1.0

BLUE = "#2166ac"
LIGHT_BLUE = "#92c5de"
RED = "#b2182b"
LIGHT_RED = "#f4a582"
GREY = "#5c6770"
LIGHT_GREY = "#d9dde1"
GREEN = "#1b7837"
GOLD = "#b8860b"


_ARTIST_BINDINGS: dict[int, list[dict[str, Any]]] = {}


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _normalise_payload(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation including array bytes."""

    if isinstance(value, np.ma.MaskedArray):
        return {
            "kind": "masked_array",
            "data": _normalise_payload(np.asarray(value.filled(np.nan))),
            "mask": _normalise_payload(np.asarray(value.mask, dtype=np.uint8)),
        }
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        if contiguous.dtype.kind == "O":
            raise TypeError("object arrays are not admissible in artist provenance")
        return {
            "kind": "array",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _normalise_payload(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_payload(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "+Inf"
        if np.isneginf(value):
            return "-Inf"
    return value


def _payload_digest(value: Any) -> str:
    encoded = json.dumps(
        _normalise_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_path(payload: Any, dotted: str) -> Any:
    value = payload
    for token in dotted.split("."):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _source_path(relative: str) -> Path:
    path = Path(relative)
    if path.parts[:2] == ("manuscript", "source_data"):
        return PROJECT / path
    return SOURCE / path


def _resolve_source_ref(ref: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Resolve an exact JSON key or NPZ key/slice before an artist is accepted."""

    path = _source_path(ref["path"])
    kind = ref["kind"]
    if kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = _json_path(payload, ref["key"])
    elif kind == "npz":
        with np.load(path, allow_pickle=False) as payload:
            value = np.asarray(payload[ref["key"]])
            if "slice" in ref:
                selector = tuple(
                    slice(None) if token == ":" else int(token)
                    for token in ref["slice"]
                )
                value = value[selector]
    else:
        raise ValueError(f"unsupported source-ref kind: {kind}")
    record = {
        **ref,
        "source_sha256": _sha256(path),
        "resolved_payload_sha256": _payload_digest(value),
    }
    return value, record


def artist_payload(artist: Any) -> dict[str, Any]:
    """Read numerical/text payload from the actual Matplotlib artist object."""

    if isinstance(artist, AxesImage):
        return {
            "type": "AxesImage",
            "array": np.asanyarray(artist.get_array()),
            "extent": [float(value) for value in artist.get_extent()],
            "clim": [float(value) for value in artist.get_clim()],
        }
    if isinstance(artist, Quiver):
        return {
            "type": "Quiver",
            "x": np.asarray(artist.X, dtype=float),
            "y": np.asarray(artist.Y, dtype=float),
            "u": np.asarray(artist.U, dtype=float),
            "v": np.asarray(artist.V, dtype=float),
        }
    if isinstance(artist, BarContainer):
        return {
            "type": "BarContainer",
            "centres": [
                float(patch.get_x() + patch.get_width() / 2) for patch in artist.patches
            ],
            "heights": [float(patch.get_height()) for patch in artist.patches],
        }
    if isinstance(artist, ErrorbarContainer):
        data_line, _, bar_collections = artist.lines
        segments = []
        for collection in bar_collections:
            segments.extend(
                np.asarray(segment, dtype=float) for segment in collection.get_segments()
            )
        if data_line is None:
            centres = np.asarray(
                [np.mean(segment, axis=0) for segment in segments],
                dtype=float,
            )
            x_data = centres[:, 0]
            y_data = centres[:, 1]
        else:
            x_data = np.asarray(data_line.get_xdata(orig=False), dtype=float)
            y_data = np.asarray(data_line.get_ydata(orig=False), dtype=float)
        return {
            "type": "ErrorbarContainer",
            "x": x_data,
            "y": y_data,
            "segments": segments,
        }
    if isinstance(artist, PathCollection):
        return {
            "type": "PathCollection",
            "offsets": np.asarray(artist.get_offsets(), dtype=float),
        }
    if isinstance(artist, Line2D):
        return {
            "type": "Line2D",
            "x": np.asarray(artist.get_xdata(orig=False), dtype=float),
            "y": np.asarray(artist.get_ydata(orig=False), dtype=float),
        }
    if isinstance(artist, Rectangle):
        # A horizontal span (``axhspan``) carries its evidence in the vertical
        # bounds only; the horizontal extent is the axes width, not data.
        _, y0 = artist.get_xy()
        return {
            "type": "Rectangle",
            "y_bounds": [float(y0), float(y0 + artist.get_height())],
        }
    if isinstance(artist, Text):
        return {"type": "Text", "text": artist.get_text()}
    raise TypeError(f"unsupported provenance artist type: {type(artist).__name__}")


def expected_hspan_payload(lower: float, upper: float) -> dict[str, Any]:
    """Payload Matplotlib should expose for ``axhspan(lower, upper)``."""

    return {"type": "Rectangle", "y_bounds": [float(lower), float(upper)]}


def expected_errorbar_payload(
    x: Any,
    y: Any,
    lower: Any,
    upper: Any,
    *,
    data_line: bool = True,
) -> dict[str, Any]:
    """Construct the payload Matplotlib should expose for vertical error bars."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    lower_values = np.asarray(lower, dtype=float).reshape(-1)
    upper_values = np.asarray(upper, dtype=float).reshape(-1)
    # Matplotlib receives asymmetric distances, then reconstructs endpoints;
    # reproduce that floating-point path rather than comparing to the original
    # endpoint bytes.
    rendered_lower = y_values - (y_values - lower_values)
    rendered_upper = y_values + (upper_values - y_values)
    return {
        "type": "ErrorbarContainer",
        "x": x_values,
        "y": (
            y_values
            if data_line
            else 0.5 * (rendered_lower + rendered_upper)
        ),
        "segments": [
            np.asarray(
                [[x_value, low], [x_value, high]],
                dtype=float,
            )
            for x_value, low, high in zip(
                x_values, rendered_lower, rendered_upper
            )
        ],
    }


def expected_bar_payload(centres: Any, heights: Any) -> dict[str, Any]:
    return {
        "type": "BarContainer",
        "centres": np.asarray(centres, dtype=float).reshape(-1).tolist(),
        "heights": np.asarray(heights, dtype=float).reshape(-1).tolist(),
    }


def bind_artist(
    fig: plt.Figure,
    artist: Any,
    *,
    artist_id: str,
    panel: str,
    source_refs: list[dict[str, Any]],
    source_payload: list[Any],
    expected_payload: dict[str, Any],
    transform: str,
    evidence: str,
) -> None:
    """Bind an actual artist to exact source keys and a declared transformation.

    Source selection occurs first. Each declared key/slice is reloaded and must
    equal the producer value before the actual Matplotlib object is inspected.
    This prevents a numerically similar but semantically unrelated key from
    satisfying the audit.
    """

    if len(source_refs) != len(source_payload):
        raise ValueError(f"{artist_id}: source refs/payload length mismatch")
    source_records = []
    for ref, expected_source in zip(source_refs, source_payload):
        resolved, record = _resolve_source_ref(ref)
        if _payload_digest(resolved) != _payload_digest(expected_source):
            raise RuntimeError(
                f"{artist_id}: declared source {ref} does not match producer payload"
            )
        source_records.append(record)
    actual = artist_payload(artist)
    if _payload_digest(actual) != _payload_digest(expected_payload):
        raise RuntimeError(
            f"{artist_id}: actual Matplotlib payload differs from source-derived expectation"
        )
    _ARTIST_BINDINGS.setdefault(id(fig), []).append(
        {
            "artist_id": artist_id,
            "panel": panel,
            "artist_type": actual["type"],
            "source_refs": source_records,
            "transform": transform,
            "evidence": evidence,
            "actual_payload_sha256": _payload_digest(actual),
            "expected_payload_sha256": _payload_digest(expected_payload),
            "validated": True,
        }
    )


def write_artist_manifest(fig: plt.Figure, stem: str) -> dict[str, Any]:
    bindings = sorted(
        _ARTIST_BINDINGS.pop(id(fig), []),
        key=lambda item: item["artist_id"],
    )
    if not bindings:
        raise RuntimeError(f"{stem}: no producer-emitted artist provenance")
    identifiers = [record["artist_id"] for record in bindings]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"{stem}: duplicate artist provenance identifiers")
    payload = {
        "schema": "gwt-producer-emitted-artist-manifest-v1",
        "figure": stem,
        "grouping": (
            "One record per semantically indivisible plotted group (image, bar series, "
            "scatter series, error-bar series or data-bearing text)."
        ),
        "binding_count": len(bindings),
        "all_validated": all(record["validated"] for record in bindings),
        "bindings": bindings,
    }
    destination = ARTIST_MANIFEST_DIR / f"{stem}.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "lines.linewidth": 1.6,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(relative: str) -> dict:
    return json.loads((SOURCE / relative).read_text(encoding="utf-8"))


def _intersection(first, second) -> tuple[float, float]:
    return (
        max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0)),
        max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0)),
    )


def rendered_layout_audit(fig: plt.Figure, stem: str) -> dict:
    """Reject clipping, text/text collisions and annotation/reference-line collisions."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tolerance_px = 0.5
    text_records = []
    overflow = []
    final_scale = (FINAL_DOUBLE_COLUMN_MM / 25.4) / fig.get_size_inches()[0]
    final_font_sizes = []
    final_font_failures = []
    for index, artist in enumerate(fig.findobj(match=Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if bounds.width <= 0 or bounds.height <= 0:
            continue
        record = {
            "index": index,
            "artist": artist,
            "text": artist.get_text(),
            "bbox": bounds,
        }
        text_records.append(record)
        final_font_pt = float(artist.get_fontsize()) * final_scale
        final_font_sizes.append(final_font_pt)
        if not (MIN_FINAL_FONT_PT <= final_font_pt <= MAX_FINAL_FONT_PT):
            final_font_failures.append(
                {
                    "text": artist.get_text(),
                    "source_font_pt": float(artist.get_fontsize()),
                    "effective_font_pt_at_180mm": final_font_pt,
                }
            )
        if (
            bounds.x0 < canvas.x0 - tolerance_px
            or bounds.y0 < canvas.y0 - tolerance_px
            or bounds.x1 > canvas.x1 + tolerance_px
            or bounds.y1 > canvas.y1 + tolerance_px
        ):
            overflow.append(
                {
                    "text": artist.get_text(),
                    "bbox": [bounds.x0, bounds.y0, bounds.x1, bounds.y1],
                }
            )

    # Nature Communications requests lines of at least 1 pt at final
    # reproduction size.  Test actual rendered line and marker-edge artists
    # after scaling the source canvas to the declared 180-mm final width.
    final_stroke_weights = []
    final_stroke_failures = []
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        # Legend proxy handles have no data axes and do not define a plotted
        # stroke; their source artist is audited on the axes itself.
        if getattr(artist, "axes", None) is None:
            continue
        widths = []
        artist_type = type(artist).__name__
        if isinstance(artist, Line2D):
            widths = [float(artist.get_linewidth())]
        elif isinstance(artist, (LineCollection, PathCollection)):
            widths = [
                float(width)
                for width in np.asarray(artist.get_linewidths(), dtype=float).ravel()
            ]
        elif isinstance(artist, Spine):
            widths = [float(artist.get_linewidth())]
        for source_width in widths:
            if source_width <= 0:
                continue
            final_width = source_width * final_scale
            final_stroke_weights.append(final_width)
            # A 0.02-pt tolerance accommodates a nominal 1-pt stroke when a
            # 182.9-mm source canvas is reproduced at 180 mm.
            if final_width + 0.02 < MIN_FINAL_STROKE_PT:
                final_stroke_failures.append(
                    {
                        "artist_type": artist_type,
                        "label": str(artist.get_label()),
                        "axes_title": (
                            artist.axes.get_title()
                            if getattr(artist, "axes", None) is not None
                            else None
                        ),
                        "axes_label": (
                            artist.axes.get_label()
                            if getattr(artist, "axes", None) is not None
                            else None
                        ),
                        "source_stroke_pt": source_width,
                        "effective_stroke_pt_at_180mm": final_width,
                    }
                )

    text_collisions = []
    for left_index, left in enumerate(text_records):
        for right in text_records[left_index + 1 :]:
            width, height = _intersection(left["bbox"], right["bbox"])
            if width <= 1.5 or height <= 1.5:
                continue
            intersection = width * height
            smaller = min(
                left["bbox"].width * left["bbox"].height,
                right["bbox"].width * right["bbox"].height,
            )
            if intersection / max(smaller, 1.0) < 0.04:
                continue
            text_collisions.append(
                {
                    "first": left["text"],
                    "second": right["text"],
                    "overlap_fraction_of_smaller": intersection / max(smaller, 1.0),
                }
            )

    reference_line_collisions = []
    for ax in fig.axes:
        axes_width = ax.get_window_extent(renderer=renderer).width
        axes_height = ax.get_window_extent(renderer=renderer).height
        annotations = [
            text for text in ax.texts if text.get_visible() and text.get_text().strip()
        ]
        for line in ax.findobj(match=Line2D):
            if not line.get_visible():
                continue
            x = np.asarray(line.get_xdata(orig=False), dtype=float)
            y = np.asarray(line.get_ydata(orig=False), dtype=float)
            if x.size < 2 or y.size < 2:
                continue
            path_bounds = line.get_path().get_extents(line.get_transform())
            horizontal = np.ptp(y) < 1e-12 and path_bounds.width >= 0.75 * axes_width
            vertical = np.ptp(x) < 1e-12 and path_bounds.height >= 0.75 * axes_height
            if not (horizontal or vertical):
                continue
            expanded = path_bounds.expanded(
                1.0 + 3.0 / max(path_bounds.width, 1.0),
                1.0 + 3.0 / max(path_bounds.height, 1.0),
            )
            for text in annotations:
                bounds = text.get_window_extent(renderer=renderer)
                width, height = _intersection(bounds, expanded)
                if width > 1.5 and height > 1.5:
                    reference_line_collisions.append(
                        {
                            "text": text.get_text(),
                            "orientation": "horizontal" if horizontal else "vertical",
                        }
                    )

    # Text whose centre lies inside a rounded information box must remain
    # inside that box.  Canvas and text/text checks alone missed overflowing
    # labels in an earlier Figure 1 release.
    patch_containment_failures = []
    for ax in fig.axes:
        patches = [
            patch
            for patch in ax.findobj(match=FancyBboxPatch)
            if patch.get_visible()
        ]
        patch_bounds = [
            (patch, patch.get_window_extent(renderer=renderer)) for patch in patches
        ]
        for text_artist in ax.texts:
            if not text_artist.get_visible() or not text_artist.get_text().strip():
                continue
            text_bounds = text_artist.get_window_extent(renderer=renderer)
            centre_x = 0.5 * (text_bounds.x0 + text_bounds.x1)
            centre_y = 0.5 * (text_bounds.y0 + text_bounds.y1)
            containers = [
                bounds
                for _, bounds in patch_bounds
                if bounds.x0 <= centre_x <= bounds.x1
                and bounds.y0 <= centre_y <= bounds.y1
            ]
            if not containers:
                continue
            container = min(containers, key=lambda item: item.width * item.height)
            padding = 1.0
            if (
                text_bounds.x0 < container.x0 + padding
                or text_bounds.x1 > container.x1 - padding
                or text_bounds.y0 < container.y0 + padding
                or text_bounds.y1 > container.y1 - padding
            ):
                patch_containment_failures.append(
                    {
                        "text": text_artist.get_text(),
                        "text_bbox": [
                            text_bounds.x0,
                            text_bounds.y0,
                            text_bounds.x1,
                            text_bounds.y1,
                        ],
                        "container_bbox": [
                            container.x0,
                            container.y0,
                            container.x1,
                            container.y1,
                        ],
                    }
                )

    report = {
        "stem": stem,
        "text_artists_checked": len(text_records),
        "canvas_overflow": overflow,
        "text_text_collisions": text_collisions,
        "text_reference_line_collisions": reference_line_collisions,
        "text_patch_containment_failures": patch_containment_failures,
        "final_width_mm": FINAL_DOUBLE_COLUMN_MM,
        "effective_font_pt_range_at_180mm": [
            min(final_font_sizes) if final_font_sizes else None,
            max(final_font_sizes) if final_font_sizes else None,
        ],
        "final_font_size_failures": final_font_failures,
        "minimum_effective_stroke_pt_at_180mm": (
            min(final_stroke_weights) if final_stroke_weights else None
        ),
        "final_stroke_failures": final_stroke_failures,
        "all_pass": not overflow
        and not text_collisions
        and not reference_line_collisions
        and not patch_containment_failures
        and not final_font_failures
        and not final_stroke_failures,
    }
    audit_directory = os.environ.get("GWT_LAYOUT_AUDIT_DIR")
    if audit_directory:
        destination = Path(audit_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{stem}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def save(fig: plt.Figure, stem: str) -> None:
    provenance = write_artist_manifest(fig, stem)
    audit = rendered_layout_audit(fig, stem)
    if not audit["all_pass"]:
        raise RuntimeError(f"{stem}: rendered layout audit failed: {audit}")

    # Do not use a tight bounding box here.  A text artist accidentally placed
    # in data rather than axes coordinates can otherwise create a many-metre
    # PDF canvas while still looking like a successful save.
    fig.savefig(OUTPUT / f"{stem}.png", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    print(
        f"LAYOUT_AUDIT: {stem} ALL_PASS "
        f"{audit['text_artists_checked']} text artists"
    )
    print(
        f"ARTIST_PROVENANCE: {stem} ALL_PASS "
        f"{provenance['binding_count']} source-first groups"
    )
    print(f"wrote {OUTPUT / f'{stem}.png'}")


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.13,
    y: float = 1.08,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
