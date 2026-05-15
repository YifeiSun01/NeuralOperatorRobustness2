# FNO nu=0.001 Target Loss And Same-Delta Loss3 By Step

Rows are averaged over the five initial conditions. Each attack path is evaluated at the same saved delta_k.

| k | optimize loss1: target loss1 (same-delta loss3, budget) | optimize loss2: target loss2 (same-delta loss3, budget) | optimize loss3: direct loss3 (budget) | direct loss3 - loss1-path loss3 | direct loss3 - loss2-path loss3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `4.9209 (L3=0.7918, b=0.323)` | `4.9880 (L3=0.2769, b=0.317)` | `0.3438 (b=0.036)` | `-0.4480` | `+0.0669` |
| 10 | `7.0218 (L3=1.6062, b=0.538)` | `6.9869 (L3=0.6956, b=0.530)` | `0.5438 (b=0.092)` | `-1.0624` | `-0.1518` |
| 15 | `8.8332 (L3=2.4386, b=0.734)` | `8.6381 (L3=1.5486, b=0.722)` | `0.9348 (b=0.171)` | `-1.5037` | `-0.6138` |
| 20 | `10.2920 (L3=3.2480, b=0.904)` | `9.9641 (L3=2.4301, b=0.892)` | `1.5692 (b=0.269)` | `-1.6788` | `-0.8609` |
| 25 | `10.9582 (L3=3.6986, b=0.992)` | `10.7516 (L3=3.1638, b=0.999)` | `2.1124 (b=0.365)` | `-1.5862` | `-1.0513` |
| 30 | `11.0809 (L3=3.8242, b=1.000)` | `10.8192 (L3=3.2967, b=1.000)` | `2.6134 (b=0.459)` | `-1.2108` | `-0.6833` |
| 35 | `11.1273 (L3=3.8887, b=1.000)` | `10.8617 (L3=3.4127, b=1.000)` | `3.1326 (b=0.554)` | `-0.7561` | `-0.2801` |
| 40 | `11.1638 (L3=3.9494, b=1.000)` | `10.8954 (L3=3.5186, b=1.000)` | `3.6236 (b=0.632)` | `-0.3258` | `+0.1050` |
| 45 | `11.1935 (L3=4.0084, b=1.000)` | `10.9228 (L3=3.6137, b=1.000)` | `4.1999 (b=0.708)` | `+0.1915` | `+0.5861` |
| 50 | `11.2179 (L3=4.0678, b=1.000)` | `10.9451 (L3=3.6982, b=1.000)` | `4.6171 (b=0.764)` | `+0.5493` | `+0.9188` |

Interpretation:

- Early and middle steps are not a clean win for direct loss3, because the loss1/loss2 paths use much more L2 budget.
- At the final saved step k=50, direct loss3 gives the largest mean loss3: 4.6171 vs 4.0678 on the loss1 path and 3.6982 on the loss2 path.
- Therefore the correct statement is not that loss1/loss2 always produce lower loss3. The cleaner statement is: after comparable late-stage attack progress, direct loss3 is the most targeted way to raise loss3, while loss1/loss2 mainly drive model output movement.
