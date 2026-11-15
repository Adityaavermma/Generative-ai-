| Run | Input · model | Accuracy (fold ± sd) | Balanced acc | Macro F1 | Macro AUC | Malignant sens / spec |
|---|---|---|---|---|---|---|
| `ensemble_L4_L2` | Ensemble L4+L2 · convnextv2-tiny 384px + lesion loss | 89.4% (±3.7%) | 88.3% | 0.885 | 0.966 | 87.3% / 94.8% |
| `final_L4` | L4 denoised + enhanced · convnextv2-tiny 384px + lesion loss | 88.8% (±4.9%) | 86.7% | 0.875 | 0.952 | 86.3% / 95.3% |
| `cmp_L2_256` | L2 denoised · convnext-tiny 256px | 88.5% (±2.4%) | 86.1% | 0.869 | 0.943 | 89.2% / 93.7% |
| `final_L2` | L2 denoised · convnextv2-tiny 384px + lesion loss | 88.3% (±3.4%) | 86.8% | 0.872 | 0.949 | 84.3% / 95.0% |
| `cmp_L0_256` | L0 raw · convnext-tiny 256px | 86.9% (±3.5%) | 82.6% | 0.844 | 0.946 | 89.2% / 92.8% |
| `cmp_L4_256` | L4 denoised + enhanced · convnext-tiny 256px | 86.1% (±1.9%) | 83.6% | 0.845 | 0.938 | 84.8% / 92.3% |
| `cmp_L4_384_seg_tiny` | L4 denoised + enhanced · convnext-tiny 384px + lesion loss | 85.7% (±2.5%) | 81.7% | 0.830 | 0.935 | 82.4% / 96.4% |

chart: C:\Users\hp\OneDrive\Desktop\ML\gen ai\ultrasound_disease_classifier\runs\comparison.png
