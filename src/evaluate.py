"""
Evaluation module for Open-Set Presentation Attack Detection.

Upgraded evaluation pipeline:
  1. Extract z0 latent + reconstruction errors from neural net
  2. Apply PCA + KMeans + per-cluster Isolation Forests on bona-fide ONLY features
  3. Also compute direct Spectral L1 anomaly score (most discriminative)
  4. Fuse: IF score + Mahalanobis + Spectral L1 + Latent MSE
  5. Calibrate threshold on validation set at BPCER <= BPCER_TARGET
  6. Apply calibrated threshold to test set for APCER/HTER computation

Key fixes vs. previous version:
  - Reconstruction errors (latent_mse, spectral_l1) are included in PCA features
  - Score normalization uses val-set limits (no test-set leakage)
  - Threshold is calibrated on val set at target BPCER operating point
  - Binary predictions use calibrated threshold (not IF contamination=0.14)
"""

import os
import json
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import mahalanobis
from scipy import ndimage
import joblib

from . import config as cfg


class AOPADEvaluator:
    """
    AOPAD-style evaluator with threshold calibration.
    """

    def __init__(self, model, device=None, noise_level=None,
                 k_max=15, contamination_range=None, pca_variance=0.95):
        self.model = model
        self.device = device or cfg.DEVICE
        self.noise_level = noise_level or cfg.NOISE_LEVEL
        self.k_max = k_max
        self.contamination_range = contamination_range or [
            round(v, 2) for v in np.arange(0.01, 0.31, 0.01)
        ]
        self.pca_variance = pca_variance

        self.model = self.model.to(self.device)
        self.model.eval()

        # Fitted during fit()
        self.pca = None
        self.kmeans = None
        self.isolation_forests = []
        self.optimal_k = None
        self.optimal_contamination = None

        # StandardScaler: fitted on bona-fide training features, applied to all sets.
        # This ensures z0, texture, and reconstruction error dims are on equal footing
        # before PCA — preventing large-valued z0 from dominating.
        self.scaler = None

        # Normalization limits fitted on val set
        self.score_limits = {}

        # Calibrated threshold (fitted on val set at BPCER_TARGET)
        self.calibrated_threshold = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(self, data_loader, desc="Extracting features"):
        """Extract z0 latents + texture + reconstruction errors for all samples."""
        all_z0, all_labels, all_users, all_categories = [], [], [], []
        all_tex = []
        all_latent_mse, all_spectral_l1 = [], []

        for dwt_concat, hh_band, labels, metadata in tqdm(data_loader, desc=desc):
            dwt_concat  = dwt_concat.to(self.device)
            hh_original = hh_band.to(self.device)

            with autocast('cuda', enabled=cfg.USE_AMP):
                nl  = torch.full((dwt_concat.size(0),), self.noise_level).to(self.device)
                out = self.model(dwt_concat, hh_original, noise_level=nl)
                z0          = out['z0']
                latent_mse  = out['latent_mse']
                spectral_l1 = out['spectral_l1']

            all_z0.append(z0.cpu().numpy())
            all_latent_mse.append(latent_mse.cpu().numpy())
            all_spectral_l1.append(spectral_l1.cpu().numpy())
            all_labels.append(labels.numpy())
            all_users.extend(list(metadata['user_id']))
            all_categories.extend(list(metadata['category']))

            # Multi-scale LBP + Fourier + Gradient from HH band
            hh_np   = hh_band.numpy()
            hh_gray = hh_np.mean(axis=1)
            batch_tex = []
            for img in hh_gray:
                features = []
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img_uint8 = (img_norm * 255).astype(np.uint8)

                for R, P in [(1, 8), (2, 16), (3, 24)]:
                    lbp  = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')
                    hist, _ = np.histogram(lbp.ravel(), bins=P+2,
                                           range=(0, P+2), density=True)
                    features.extend(hist)

                f_transform = np.fft.fft2(img)
                f_shift     = np.fft.fftshift(f_transform)
                magnitude   = np.abs(f_shift)
                h, w = img.shape
                y, x = np.indices((h, w))
                center = (h // 2, w // 2)
                r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
                tbin = np.bincount(r.ravel(), magnitude.ravel())
                nr   = np.bincount(r.ravel())
                radial = tbin / np.maximum(nr, 1)
                if len(radial) >= 5:
                    fourier_bins = np.array_split(radial, 5)
                    fourier_f    = [np.mean(b) for b in fourier_bins]
                else:
                    fourier_f = [0.0] * 5
                f_sum = sum(fourier_f) + 1e-8
                features.extend([f / f_sum for f in fourier_f])

                sx  = ndimage.sobel(img, axis=0, mode='constant')
                sy  = ndimage.sobel(img, axis=1, mode='constant')
                sob = np.hypot(sx, sy)
                features.extend([np.mean(sob), np.std(sob)])
                batch_tex.append(features)

            all_tex.append(np.array(batch_tex))

        z0_arr         = np.concatenate(all_z0)
        tex_arr        = np.concatenate(all_tex)
        latent_mse_arr = np.concatenate(all_latent_mse)
        spectral_l1_arr = np.concatenate(all_spectral_l1)

        # Include reconstruction errors in the feature vector for PCA/IF
        combined = np.concatenate([
            z0_arr,
            tex_arr,
            latent_mse_arr[:, None],
            spectral_l1_arr[:, None],
        ], axis=1)

        return {
            'z0':          combined,
            'latent_mse':  latent_mse_arr,
            'spectral_l1': spectral_l1_arr,
            'labels':      np.concatenate(all_labels),
            'users':       np.array(all_users),
            'categories':  all_categories,
        }

    # ------------------------------------------------------------------
    # Silhouette-based K selection
    # ------------------------------------------------------------------

    def _select_optimal_k(self, X_pca):
        best_k, best_score = 2, -1.0
        for k in range(2, self.k_max + 1):
            km     = KMeans(n_clusters=k, random_state=cfg.SEED, n_init=10)
            lbl_km = km.fit_predict(X_pca)
            if len(np.unique(lbl_km)) < 2:
                continue
            score = silhouette_score(X_pca, lbl_km)
            print(f"    K={k:2d}  silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
        print(f"  -> Optimal K = {best_k}  (silhouette={best_score:.4f})")
        return best_k

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, train_loader, save_dir=None):
        """Fit PCA + KMeans + per-cluster Isolation Forests on bona-fide training data."""
        print("\n" + "=" * 60)
        print("AOPAD: FITTING ONE-CLASS ENSEMBLE")
        print("=" * 60)

        feats  = self._extract_features(train_loader, desc="[Fit] Extracting train z0")
        bf_mask = feats['labels'] == 0
        X_raw   = feats['z0'][bf_mask]
        print(f"  Bona-fide samples for fitting: {X_raw.shape[0]}")

        # --- StandardScaler: fit on bona-fide features ---
        # Normalizes each feature dimension to zero-mean unit-variance.
        # This prevents z0 dims (range ~[-3,3]) from dwarfing reconstruction
        # error dims (range ~[0.00001, 0.05]) in PCA.
        print(f"\n[0/3] Fitting StandardScaler on bona-fide features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        print(f"  Feature dims: {X_raw.shape[1]}  |  Scaler fitted on {X_raw.shape[0]} BF samples")

        print(f"\n[1/3] Applying PCA (variance={self.pca_variance})...")
        self.pca   = PCA(n_components=self.pca_variance, random_state=cfg.SEED)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"  PCA: {X_scaled.shape[1]}D -> {X_pca.shape[1]}D")

        print(f"\n[2/3] Selecting optimal K (k_max={self.k_max})...")
        self.optimal_k = self._select_optimal_k(X_pca)
        self.kmeans    = KMeans(n_clusters=self.optimal_k,
                                random_state=cfg.SEED, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca)

        FIXED_CONTAMINATION = cfg.CONTAMINATION
        print(f"\n[3/3] Fitting {self.optimal_k} Isolation Forests "
              f"(contamination={FIXED_CONTAMINATION})...")
        self.isolation_forests = []
        contaminations         = []
        self.cluster_means     = []
        self.cluster_cov_inv   = []

        for k in range(self.optimal_k):
            mask_k = cluster_labels == k
            X_k    = X_pca[mask_k]
            iso    = IsolationForest(contamination=FIXED_CONTAMINATION,
                                     random_state=cfg.SEED, n_jobs=-1)
            iso.fit(X_k)
            self.isolation_forests.append(iso)
            contaminations.append(FIXED_CONTAMINATION)

            mu     = np.mean(X_k, axis=0)
            cov    = np.cov(X_k, rowvar=False) + np.eye(X_k.shape[1]) * 1e-6
            cov_inv = np.linalg.inv(cov)
            self.cluster_means.append(mu)
            self.cluster_cov_inv.append(cov_inv)
            print(f"  Cluster {k}: n={mask_k.sum()}, contamination={FIXED_CONTAMINATION:.2f}")

        self.optimal_contamination = contaminations
        print("\n  [v] AOPAD ensemble fitted.")

        if save_dir:
            self._save_ensemble(save_dir)
        return self

    # ------------------------------------------------------------------
    # Binary prediction from z0
    # ------------------------------------------------------------------

    def _predict_from_z0(self, Z_raw):
        X_scaled = self.scaler.transform(Z_raw)
        X_pca = self.pca.transform(X_scaled)
        cluster_assignments = self.kmeans.predict(X_pca)
        predictions = np.ones(len(Z_raw), dtype=int)
        for k, iso in enumerate(self.isolation_forests):
            mask_k = cluster_assignments == k
            if mask_k.sum() == 0:
                continue
            preds_k = iso.predict(X_pca[mask_k])
            predictions[mask_k] = np.where(preds_k == 1, 0, 1)
        return predictions

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_scores(self, data_loader, fit_limits=False):
        """
        Compute fused anomaly scores.
        fit_limits=True  -> fit normalization on this set (use for val)
        fit_limits=False -> reuse val-set normalization limits (use for test)
        """
        feats       = self._extract_features(data_loader, desc="Computing scores")
        Z_raw       = feats['z0']
        latent_mse  = feats['latent_mse']
        spectral_l1 = feats['spectral_l1']

        # Apply scaler (fitted on bona-fide training data) then PCA
        X_scaled            = self.scaler.transform(Z_raw)
        X_pca               = self.pca.transform(X_scaled)
        cluster_assignments  = self.kmeans.predict(X_pca)

        # 1. IF score
        all_iso = np.zeros((len(Z_raw), len(self.isolation_forests)))
        for k, iso in enumerate(self.isolation_forests):
            all_iso[:, k] = -iso.decision_function(X_pca)
        s_iso = all_iso.mean(axis=1)

        # 2. Mahalanobis
        s_maha = np.zeros(len(Z_raw))
        for i in range(len(Z_raw)):
            x     = X_pca[i]
            dists = [mahalanobis(x, self.cluster_means[k], self.cluster_cov_inv[k])
                     for k in range(self.optimal_k)]
            s_maha[i] = min(dists)

        # 3. Latent MSE
        s_lmse = latent_mse

        # 4. Spectral L1
        s_spec = spectral_l1

        # Normalise using val-set limits (no test leakage).
        # Percentile clipping (1%/99%) is used instead of hard min/max to
        # avoid a single outlier attack score distorting the entire scale.
        def _norm01(x, name):
            if fit_limits or name not in self.score_limits:
                lo = float(np.percentile(x, 1))
                hi = float(np.percentile(x, 99))
                if hi == lo:
                    hi = lo + 1e-8
                self.score_limits[name] = (lo, hi)
            else:
                lo, hi = self.score_limits[name]
            return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        s_iso_n   = _norm01(s_iso,   'iso')
        s_maha_n  = _norm01(s_maha,  'maha')
        s_lmse_n  = _norm01(s_lmse,  'lmse')
        s_spec_n  = _norm01(s_spec,  'spec')

        s_final = (cfg.W_ISO  * s_iso_n  +
                   cfg.W_MAHA * s_maha_n +
                   cfg.W_LMSE * s_lmse_n +
                   cfg.W_SPEC * s_spec_n)

        # Binary predictions: use IF ensemble for APCER/BPCER computation
        predictions = self._predict_from_z0(Z_raw)

        return {
            's_final':     s_final,
            's_spec':      s_spec,          # raw spectral l1 (for threshold calib)
            'predictions': predictions,
            'labels':      feats['labels'],
            'users':       feats['users'],
            'categories':  feats['categories'],
        }

    # ------------------------------------------------------------------
    # Threshold calibration (val set -> BPCER_TARGET)
    # ------------------------------------------------------------------

    def calibrate_threshold(self, val_scores):
        """
        Set self.calibrated_threshold so that BPCER on val set == BPCER_TARGET.
        Uses the continuous s_final score.

        Score Inversion Guard:
          If attack scores are LOWER than bona-fide scores on the val set
          (i.e., the model reconstructs attacks better than bona-fide),
          we flip s_final = 1 - s_final so that higher score = more anomalous.
          This is a safety net; after successful retraining it should not trigger.
        """
        bpcer_target = getattr(cfg, 'BPCER_TARGET', 0.10)
        labels  = val_scores['labels']
        s_final = val_scores['s_final']
        bf_mask  = (labels == 0)
        atk_mask = (labels > 0)

        if bf_mask.sum() == 0:
            self.calibrated_threshold = 0.5
            return

        # --- Separation diagnostic ---
        bf_mean  = float(s_final[bf_mask].mean())
        bf_std   = float(s_final[bf_mask].std()) + 1e-8
        atk_mean = float(s_final[atk_mask].mean()) if atk_mask.sum() > 0 else bf_mean
        sep_ratio = (atk_mean - bf_mean) / bf_std
        print(f"  Score diagnostic:")
        print(f"    Bona-fide mean_score = {bf_mean:.4f}  (+/- {bf_std:.4f})")
        print(f"    Attack    mean_score = {atk_mean:.4f}")
        print(f"    Separation ratio     = {sep_ratio:+.3f} std  "
              f"({'GOOD: attacks > bona-fide' if sep_ratio > 0 else 'WARNING: scores INVERTED'})")

        # --- Inversion guard ---
        self.score_inverted = False
        if sep_ratio < 0:
            print(f"  [!] Score inversion detected (sep={sep_ratio:.3f}). "
                  f"Flipping s_final = 1 - s_final.")
            val_scores['s_final'] = 1.0 - s_final
            s_final = val_scores['s_final']
            self.score_inverted = True

        bf_scores = np.sort(s_final[bf_mask])[::-1]   # descending
        idx = int(np.ceil(bpcer_target * len(bf_scores)))
        idx = min(idx, len(bf_scores) - 1)
        self.calibrated_threshold = float(bf_scores[idx])
        print(f"  Calibrated threshold: {self.calibrated_threshold:.6f} "
              f"(BPCER_target={bpcer_target*100:.0f}%, "
              f"score_inverted={self.score_inverted})")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, scores, use_calibrated_threshold=True):
        """Compute APCER, BPCER, ACER, HTER, AUC-ROC, EER."""
        labels    = scores['labels']
        s_final   = scores['s_final']
        # Apply same score inversion to test set if val set was inverted
        if getattr(self, 'score_inverted', False):
            s_final = 1.0 - s_final
        labels_binary = (labels > 0).astype(int)
        bf_mask   = labels == 0
        atk_mask  = labels > 0

        # Use calibrated threshold for binary predictions
        if use_calibrated_threshold and self.calibrated_threshold is not None:
            predictions = (s_final > self.calibrated_threshold).astype(int)
        else:
            predictions = scores['predictions']

        bpcer = float(predictions[bf_mask].mean())  if bf_mask.sum()  > 0 else 0.0
        apcer = float(1.0 - predictions[atk_mask].mean()) if atk_mask.sum() > 0 else 0.0
        acer  = (apcer + bpcer) / 2.0
        hter  = acer

        if len(np.unique(labels_binary)) > 1:
            auc_roc = roc_auc_score(labels_binary, s_final)
            fpr, tpr, _ = roc_curve(labels_binary, s_final)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        else:
            auc_roc = 0.0
            eer = 0.0

        accuracy = (predictions == labels_binary).mean()

        metrics = {
            'overall': {
                'APCER':    float(apcer),
                'BPCER':    float(bpcer),
                'ACER':     float(acer),
                'HTER':     float(hter),
                'AUC_ROC':  float(auc_roc),
                'EER':      float(eer),
                'accuracy': float(accuracy),
                'n_bona_fide': int(bf_mask.sum()),
                'n_attack':    int(atk_mask.sum()),
            },
            'per_attack': {},
        }

        for attack_label in [1, 2, 3, 4]:
            attack_name = cfg.ATTACK_NAMES[attack_label]
            type_mask   = labels == attack_label
            if type_mask.sum() == 0:
                continue

            type_apcer = 1.0 - predictions[type_mask].mean()
            is_known   = attack_label in cfg.KNOWN_ATTACK_TYPES
            combined   = bf_mask | type_mask
            y_true_t   = labels_binary[combined]
            y_score_t  = s_final[combined]

            if len(np.unique(y_true_t)) > 1:
                type_auc = roc_auc_score(y_true_t, y_score_t)
                fpr_t, tpr_t, _ = roc_curve(y_true_t, y_score_t)
                fnr_t = 1 - tpr_t
                ei = np.nanargmin(np.abs(fpr_t - fnr_t))
                type_eer = float((fpr_t[ei] + fnr_t[ei]) / 2)
            else:
                type_auc = 0.0
                type_eer = 0.0

            metrics['per_attack'][attack_name] = {
                'APCER':      float(type_apcer),
                'AUC_ROC':    float(type_auc),
                'EER':        float(type_eer),
                'mean_score': float(s_final[type_mask].mean()),
                'std_score':  float(s_final[type_mask].std()),
                'n_samples':  int(type_mask.sum()),
                'known_attack': is_known,
            }

        if bf_mask.sum() > 0:
            metrics['bona_fide_stats'] = {
                'mean_score': float(s_final[bf_mask].mean()),
                'std_score':  float(s_final[bf_mask].std()),
                'n_samples':  int(bf_mask.sum()),
            }

        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_evaluation(self, train_loader, val_loader, test_loader,
                        save_dir=None):
        """Complete AOPAD evaluation pipeline with threshold calibration."""
        if save_dir is None:
            save_dir = cfg.RESULTS_DIR
        os.makedirs(save_dir, exist_ok=True)

        self.fit(train_loader, save_dir=save_dir)

        print("\n" + "=" * 60)
        print("OPEN-SET EVALUATION (AOPAD)")
        print("=" * 60)

        print("\n[1/2] Evaluating on validation set (fit normalization limits)...")
        val_scores  = self.compute_scores(val_loader, fit_limits=True)
        self.calibrate_threshold(val_scores)
        val_metrics = self.compute_metrics(val_scores, use_calibrated_threshold=True)

        print("[2/2] Evaluating on test set (using val-set limits & threshold)...")
        test_scores  = self.compute_scores(test_loader, fit_limits=False)
        test_metrics = self.compute_metrics(test_scores, use_calibrated_threshold=True)

        self._print_results(test_metrics, val_metrics)

        results = {
            'optimal_k':             self.optimal_k,
            'optimal_contamination': self.optimal_contamination,
            'pca_components':        int(self.pca.n_components_),
            'calibrated_threshold':  self.calibrated_threshold,
            'val_metrics':           val_metrics,
            'test_metrics':          test_metrics,
        }

        path = os.path.join(save_dir, 'evaluation_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {path}")

        np.savez(os.path.join(save_dir, 'val_scores.npz'),
                 s_final=val_scores['s_final'],
                 labels=val_scores['labels'])
        np.savez(os.path.join(save_dir, 'test_scores.npz'),
                 s_final=test_scores['s_final'],
                 labels=test_scores['labels'])

        return results, val_scores, test_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_ensemble(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.pca,               os.path.join(save_dir, 'pca.pkl'))
        joblib.dump(self.kmeans,            os.path.join(save_dir, 'kmeans.pkl'))
        joblib.dump(self.isolation_forests, os.path.join(save_dir, 'iso_forests.pkl'))
        joblib.dump(self.scaler,            os.path.join(save_dir, 'scaler.pkl'))
        np.savez(os.path.join(save_dir, 'covariances.npz'),
                 means=self.cluster_means,
                 cov_invs=self.cluster_cov_inv)
        print(f"  Ensemble saved to {save_dir}")

    def load_ensemble(self, save_dir):
        self.pca               = joblib.load(os.path.join(save_dir, 'pca.pkl'))
        self.kmeans            = joblib.load(os.path.join(save_dir, 'kmeans.pkl'))
        self.isolation_forests = joblib.load(os.path.join(save_dir, 'iso_forests.pkl'))
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print("  [Warning] scaler.pkl not found — re-fit ensemble to generate it.")
        covs = np.load(os.path.join(save_dir, 'covariances.npz'))
        self.cluster_means   = list(covs['means'])
        self.cluster_cov_inv = list(covs['cov_invs'])
        self.optimal_k       = self.kmeans.n_clusters
        print(f"  Ensemble loaded from {save_dir}")
        return self

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def _print_results(self, test_metrics, val_metrics):
        print(f"\n{'=' * 70}")
        print("TEST RESULTS  (AOPAD - calibrated threshold)")
        print(f"{'=' * 70}")
        print(f"\n  Paper target (Table IV): ACER=0.160  BPCER=0.138  APCER=0.183  "
              f"HTER=0.041  EER=0.034")
        print(f"  {'-'*66}")

        o = test_metrics['overall']
        print(f"\n  Overall (our model):")
        print(f"    ACER:     {o['ACER']*100:.3f}%  (paper: 16.0%)")
        print(f"    BPCER:    {o['BPCER']*100:.3f}%  (paper: 13.8%)")
        print(f"    APCER:    {o['APCER']*100:.3f}%  (paper: 18.3%)")
        print(f"    HTER:     {o['HTER']*100:.3f}%  (paper:  4.1%)")
        print(f"    EER:      {o['EER']*100:.3f}%  (paper:  3.4%)")
        print(f"    AUC-ROC:  {o['AUC_ROC']:.4f}")
        print(f"    Accuracy: {o['accuracy']*100:.2f}%")

        print(f"\n  Per Attack Type:")
        print(f"  {'Type':<24} {'APCER':>8} {'AUC':>8} {'EER':>8} {'Known?':>8} {'N':>6}")
        print(f"  {'-'*60}")
        for name, m in test_metrics['per_attack'].items():
            known_str = "Yes (K)" if m['known_attack'] else "No (U)"
            print(f"  {name:<24} {m['APCER']*100:>7.2f}% "
                  f"{m['AUC_ROC']:>7.4f} {m['EER']*100:>7.2f}% "
                  f"{known_str:>8} {m['n_samples']:>6}")

        if 'bona_fide_stats' in test_metrics:
            bf = test_metrics['bona_fide_stats']
            print(f"\n  Bona Fide: mean_score={bf['mean_score']:.6f} "
                  f"+/- {bf['std_score']:.6f} (n={bf['n_samples']})")

        v = val_metrics['overall']
        print(f"\n  Validation Overall:")
        print(f"    ACER: {v['ACER']*100:.3f}%   HTER: {v['HTER']*100:.3f}%   "
              f"EER: {v['EER']*100:.3f}%")
        print(f"{'=' * 70}")


# Backward compat alias
Evaluator = AOPADEvaluator
