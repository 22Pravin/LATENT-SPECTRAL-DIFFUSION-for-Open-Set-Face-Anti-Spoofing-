# Results Analysis: Current Model vs Base Paper

This document breaks down the results currently generated in the `results/` directory (specifically `evaluation_results.json`) and compares them directly with the performance metrics published in the base paper *"Autoencoders for Open-Set Presentation Attack Detection"*.

## 1. Overall Performance Comparison (RECOD-MPAD Dataset)

The base paper measures the framework's effectiveness using **EER (Equal Error Rate)** and **HTER (Half Total Error Rate)**. The table below compares the paper's reported results against our current model's test performance:

| Metric | Base Paper (Reported) | Our Current Model (Test Set) | Difference |
| :--- | :--- | :--- | :--- |
| **EER** | 0.034 (3.4%) | 0.237 (23.7%) | Our model is underperforming by ~20.3% |
| **HTER** | 0.041 (4.1%) | 0.208 (20.8%) | Our model is underperforming by ~16.7% |
| **ACER** | *Not explicitly listed* | 0.208 (20.8%) | Average Classification Error Rate |
| **BPCER**| *Not explicitly listed* | 0.063 (6.3%) | Bona-fide Error (False Rejection) |
| **APCER**| *Not explicitly listed* | 0.353 (35.3%) | Attack Error (False Acceptance) |
| **AUC-ROC** | *Not explicitly listed* | 0.847 (84.7%) | - |
| **Accuracy**| *Not explicitly listed* | 0.678 (67.8%) | - |

**High-Level Takeaway**: Our current implementation is severely underperforming compared to the base paper. While the paper claims an impressive 3.4% Equal Error Rate (indicating extreme robustness against all unseen attacks), our model sits at a 23.7% EER. 

### What do the ISO Metrics (ACER, BPCER, APCER) tell us?
* **BPCER (6.3%)**: The "Bona Fide Presentation Classification Error Rate" measures usability. It means the system incorrectly rejected 6.3% of real, genuine users. While not perfect, this is somewhat acceptable for a security system.
* **APCER (35.3%)**: The "Attack Presentation Classification Error Rate" measures security. It means the system **incorrectly accepted 35.3% of all attacks as genuine users**. This is a massive security flaw. 
* **ACER (20.8%)**: The Average Classification Error Rate is simply the average of BPCER and APCER (HTER is calculated similarly). Because our APCER is so high, it drags the entire average down.

The takeaway from these metrics is that **the model's security threshold is far too loose**. It is letting more than a third of attackers right through the front door.

## 2. Why is our model underperforming? (Deep Dive into Per-Attack Metrics)

By looking at the detailed breakdown of our test metrics, we can pinpoint exactly *where* the model is failing. The Open-Set protocol tests the model against attacks it has never seen. 

Here is our model's performance separated by attack type:

| Attack Type | Status | EER | APCER (Attack Presentation Classification Error Rate) | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- |
| **Print (Indoor)** | Known (Repulsion Loss) | 0.28% | 0.09% | 99.98% |
| **Print (Outdoor)**| Unknown (Unseen) | 1.17% | 0.76% | 99.81% |
| **Screen (CCE TV)**| Unknown (Unseen) | **33.5%** | **58.3%** | 75.2% |
| **Screen (HP Monitor)**| Unknown (Unseen) | **39.6%** | **82.1%** | 64.1% |

### Key Observations:
1. **Exceptional Performance on Print Attacks**: The model easily detects printed photo attacks. Even when facing the completely unseen "Print (Outdoor)" attack, the EER is an incredibly low 1.17%. The Compact-Repulsion loss pushing away the "Print (Indoor)" samples during training successfully taught the model to generalize against paper/print artifacts.
2. **Catastrophic Failure on Screen Attacks**: The model is completely blind to screen replay attacks. It misclassifies 58% of CCE TV attacks and a massive 82% of HP Monitor attacks as "bona-fide" (real faces). 

## 3. Conclusion and Next Steps

**The Discrepancy:**
The base paper achieves a 3.4% EER because it successfully learns a latent boundary that excludes *all* attack types. Our model, however, has only learned to exclude *print* artifacts. The spectral reconstruction and latent MSE are not detecting the moiré patterns or pixel grids associated with the screen attacks.

**Potential Fixes to Optimize the Model:**
1. **Adjust the Isolation Forest Fusion (`LAMBDA_FUSION`)**: The anomaly score relies on fusing Latent MSE and Spectral L1. We might need to heavily increase the weight of the Spectral L1 loss (`W_SPEC`), as screen artifacts (moiré patterns) live almost exclusively in the high-frequency (HH) wavelet band.
2. **Tweak the Compact-Repulsion Margin**: The `DELTA` and `EPSILON` hyperparameters might be over-optimizing the latent space to push away "Print" characteristics, inadvertently leaving a massive "blind spot" where Screen attacks reside near the bona-fide centroid.
3. **Contamination Parameter**: The base paper explicitly states they achieved their results using a contamination parameter of `0.14` (obtained via silhouette score). Our `evaluation_results.json` shows we used `0.05`. We should revert this hyperparameter in `src/config.py` back to `0.14`.
