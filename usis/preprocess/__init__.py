"""Composable, provenance-emitting preprocessing, and the search over pipelines.

Preprocessing is a choice with consequences, so it is recorded and its sensitivity reported -
never silently baked into the data. The M2 criterion for this layer is that **the recorded chain
reproduces the spectrum**, which is checked by replaying the record rather than the objects in
memory; see :mod:`usis.preprocess.pipeline`.

| Module | What it holds |
| --- | --- |
| `base` | One step, its record, and the reconstruction that makes a chain replayable |
| `baseline` | Three baseline estimators, each stating what it does to a peak |
| `smoothing` | Savitzky-Golay and the boxcar, both specified in samples, not cm-1 |
| `scale` | Cropping, resampling, and normalisations that say what they destroy |
| `spikes` | Despiking, and why it is the most dangerous step here (`F03`) |
| `pipeline` | The composition, and `verify_reproduction` |
| `search` | The pipeline search, and N8's sensitivity number |

Milestone: M2 (C4).
"""
