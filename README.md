# PAR<sup>2</sup>Net: End-to-end Panoramic Image Reflection Removal (TPAMI 2023)

---

### Network architecture

![fig_arch](./imgs/pipeline.jpg)

## Testing

1. Prepare data at `./data`, each group of data should contain a panoramic image `color.jpg` and a mask image `mask.jpg` (glass regions at value `1`, others at `0`)

2. Download the [pre-trained model](https://pan.baidu.com/s/155oWKoTTrdwhDBb3VsHZrA?pwd=PANO) at `./model`

3. Run the test code `python test.py`