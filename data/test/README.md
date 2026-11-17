# SCW-Pose test set

This directory contains the test split only: **118 images, 239 walnut instances, and two endpoints (tip and base) per instance**.

- `images/`: 118 RGB PNG images. Compression is lossless; the decoded pixels are identical to the source images.
- `labels/`: one YOLO pose `.txt` file per image, with the same filename stem. Each nonempty line describes one walnut.
- `manifest.json`: relative image and label filenames, image width and height, instance counts, and SHA-256 hashes of the released files.

Each annotation row contains 11 values:

```text
class cx cy width height tip_x tip_y tip_visibility base_x base_y base_visibility
```

`class` is `0` (walnut). The bounding box uses center coordinates and width/height. All x coordinates and box widths are normalized by image width; all y coordinates and box heights are normalized by image height. The two endpoints are ordered **tip, then base**. Every endpoint in this test set has visibility value `2` (visible and annotated).
