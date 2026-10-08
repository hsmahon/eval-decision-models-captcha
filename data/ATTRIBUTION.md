# Image attribution

Grid cells in `images/*.png` are square crops of photographs from **COCO 2017
(Common Objects in Context)**, obtained via the Hugging Face mirror
`detection-datasets/coco` (train shards 0–3, snapshot `cf0b223`).

- COCO images are Flickr photos assembled for research; per-image licenses
  vary by photographer (see `http://cocodataset.org/#termsofuse`). This
  private benchmark repo uses cropped regions for non-commercial model
  evaluation only.
- COCO annotations (bounding boxes) © COCO Consortium, CC BY 4.0.
- No images were taken from any CAPTCHA provider.
- Cell crops were resized to 320 px and re-composed into fixed grids.
  Short pools (fire hydrant, stop sign) reuse instances with
  horizontal-flip / context-margin variants.
