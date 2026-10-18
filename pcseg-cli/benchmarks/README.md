# Benchmarks and accuracy evaluation

Scripts behind the runtime table and the accuracy evaluation of the SoftwareX
article. They use an installed `pcseg` (with the segmentation backends) but
are not part of the package itself.

| Script | Purpose |
|---|---|
| `benchmark.py` | Runtime per stage and peak GPU memory of `run` / `auto-segment` / `pick-segment` |
| `evaluate.py` | Point-level IoU / precision / recall of a result against a hand-drawn ground truth |
| `make_demo_subset.py` | The subsampled demo copy of the example scan published on Zenodo |
| `results/` | Measured results (JSON), one file per machine |

## Runtime

```bash
python benchmarks/benchmark.py --scan outdoor.e57 --work-dir /tmp/pcseg-bench \
    --results benchmarks/results/<machine>.json [--cpu]
```

Every configuration (see `CONFIGS` in the script) runs in a fresh process,
after one untimed warm-up load, so all runs read the scan from a warm disk
cache. Each run's log is split into load / model loading / segmentation /
export time; the per-run logs and outputs stay in `--work-dir`. Don't run
anything else on the machine meanwhile. `--cpu` adds a CPU-only run at
decimation 8 (slow).

## Accuracy

Because a structured scan has exactly one point per panorama pixel, the
ground truth is drawn as a label image over the panorama - that IS a
per-point 3D labelling, with no rendering or point matching involved.

1. Export the panorama at the evaluation decimation (the article uses 4):

   ```bash
   python benchmarks/evaluate.py panorama --scan outdoor.e57 --decimation 4 --out pano.png
   ```

2. Paint a single-channel 8-bit label image of exactly the same size, e.g. as
   layers in GIMP, or with polygons in CVAT/LabelMe exported as an index mask:
   pixel value = class id, 0 = none of the evaluated classes, 255 = ignore
   (ambiguous, or not annotated). Where objects overlap, paint the smaller one
   (a window in a facade is `window`, not `building`) - pcseg resolves
   overlaps the same way. Don't use SAM-assisted annotation: the reference
   would then share the model under evaluation. Save the class names as JSON:

   ```json
   {"1": "window", "2": "building", "3": "road", "4": "curb"}
   ```

   Rough effort for the example scan at decimation 4 (1194 x 3048 px):
   building facades, road and curbs are a few large polygons (under an hour);
   the windows (well over a hundred) are most of the work (roughly 2-4 hours by
   hand). Labelling only a representative part and marking the rest 255
   is a valid, cheaper alternative.

3. Run pcseg with an `.e57` output at the same decimation and score it:

   ```bash
   pcseg run --input outdoor.e57 --decimation 4 \
       --prompt "window, building, road, curb" --auto-coverage \
       --dino-box-threshold 0.35 --dino-text-threshold 0.35 --sam-score-threshold 0.90 \
       --out result.e57
   python benchmarks/evaluate.py score --scan outdoor.e57 --decimation 4 \
       --ground-truth gt.png --gt-classes gt_classes.json --prediction result.e57 \
       --json benchmarks/results/accuracy_d4.json
   ```

`tests/test_benchmark_evaluate.py` checks `evaluate.py` on a synthetic scan
where the right answer is known.
