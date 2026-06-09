# PVD experiments summary

Cover image shape: (512, 512, 3)

## Experiment 1 — maximum payload
* payload_bytes: 166259
* payload_bits: 1330072
* psnr_db: 31.590483569196003
* mse: 45.08499272664388
* rmse: 6.71453592786902
* ssim: 0.8990791165384145
* capacity_bpp: 5.073822021484375
* pairs_used: 325662
* pairs_total: 393216

## Experiment 2 — varying payload
| fraction | bytes | PSNR (dB) | SSIM | bpp |
|---|---|---|---|---|
| 0.10 | 16625 | 47.33 | 0.9952 | 0.5074 |
| 0.25 | 41564 | 42.41 | 0.9871 | 1.2684 |
| 0.50 | 83129 | 38.89 | 0.9705 | 2.5369 |
| 0.75 | 124694 | 33.79 | 0.9303 | 3.8054 |
| 1.00 | 166259 | 31.59 | 0.8991 | 5.0738 |

## Experiment 3 — robustness
| attack | BER | PSNR(stego↔att) | exact |
|---|---|---|---|
| none (clean stego) | 0.0000 | inf | True |
| brightness +5 | 0.0008 | 34.19 | False |
| Gaussian noise sigma=2 | 0.4999 | 44.13 | False |
| salt-and-pepper 0.5% | 0.4999 | 27.56 | False |
| JPEG quality 90 | 0.4993 | 34.93 | False |
| JPEG quality 70 | 0.4999 | 32.49 | False |