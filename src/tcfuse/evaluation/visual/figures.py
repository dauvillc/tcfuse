"""Target | Prediction | Diff figures comparing several models' FIELD reconstructions.

For each of a capped number of windows shared by every model in the comparison,
renders one figure per FIELD-kind target source: one row per channel of
Target | Pred | Diff panels repeated for each model (see
:func:`tcfuse.data.visualization.comparison_fields.plot_field_prediction_comparison`).
SCALAR / PROFILE targets (e.g. best-track scalars) are skipped — the map-based
comparison only makes sense for 2-D fields.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from tcfuse.data.predictions.run import PredictionRun
from tcfuse.data.predictions.sample import SamplePrediction
from tcfuse.data.sources.source import SourceKind
from tcfuse.data.visualization.comparison_fields import plot_field_prediction_comparison
from tcfuse.data.visualization.training import field_display
from tcfuse.evaluation.base import Evaluation


class VisualEvaluation(Evaluation):
    """Per-window Target | Pred | Diff figures comparing several models.

    For each shared window (a sample id present in every model's run), and for
    every FIELD-kind target source in that window, draws one figure with a row
    per channel: the target, then each model's (prediction, diff) panel pair.

    Args:
        max_samples: Maximum number of shared windows to render (in the first
            model's index order). ``None`` renders every shared window, which
            can produce a very large number of SVGs.
    """

    name = "visual"

    def __init__(self, max_samples: int | None = 8) -> None:
        self.max_samples = max_samples

    def run(self, runs: dict[str, PredictionRun], output_dir: Path) -> None:
        """Render comparison figures for the windows shared by every model."""
        # Only windows present in every model's run can be compared side by side.
        sample_ids = self._common_sample_ids(runs)
        if self.max_samples is not None:
            sample_ids = sample_ids[: self.max_samples]

        model_names = list(runs.keys())
        for sample_id in sample_ids:
            # Random-access load this one window from every model's run.
            samples = {name: runs[name].load_sample(sample_id) for name in model_names}
            self._render_sample(samples, model_names, output_dir)

        print(f"  [{self.name}] wrote figures for {len(sample_ids)} sample(s) to {output_dir}")

    def _render_sample(
        self,
        samples: dict[str, SamplePrediction],
        model_names: list[str],
        output_dir: Path,
    ) -> None:
        """Render one figure per FIELD target source in a single shared window."""
        # Ground truth is identical across models for a shared window; read it
        # from the first model's sample.
        first_sample = samples[model_names[0]]
        for key, target_source in first_sample.target.items():
            # The mesh-based comparison only applies to 2-D FIELD sources.
            if target_source.kind is not SourceKind.FIELD:
                continue
            # Defensive: skip if any model is missing this source (shouldn't
            # happen when all models were run on the same windows setup).
            if not all(key in samples[name].predicted for name in model_names):
                continue

            source_name, source_index = key
            predictions = {name: samples[name].predicted[key].values for name in model_names}
            cmap_key, unit = field_display(source_name)
            # FIELD coords are (H, W, 2) = [lat, lon] per pixel.
            lats = target_source.coords[..., 0]
            lons = target_source.coords[..., 1]

            save_path = output_dir / f"{first_sample.sample_id}_{source_name}_{source_index}"
            fig, _axes = plot_field_prediction_comparison(
                target_source.values,
                predictions,
                lats,
                lons,
                channels=target_source.channels,
                cmap_key=cmap_key,
                unit=unit,
                mask=target_source.mask,
                suptitle=f"{source_name}[{source_index}] — {first_sample.sample_id}",
                save_path=save_path,
            )
            # Release the figure immediately; many windows/sources are rendered.
            plt.close(fig)

    @staticmethod
    def _common_sample_ids(runs: dict[str, PredictionRun]) -> list[str]:
        """Return sample ids present in every run, in the first run's order."""
        model_names = list(runs.keys())
        first_ids = runs[model_names[0]].sample_ids
        common = set(first_ids)
        for name in model_names[1:]:
            common &= set(runs[name].sample_ids)
        return [sample_id for sample_id in first_ids if sample_id in common]
