import numpy as np
import matplotlib.pyplot as plt

result = np.load("PINN_training_logs.npz", allow_pickle=True)
training_loss = result["train_losses"].astype(float)
validation_loss = result["val_total"].astype(float)
pde_loss = result["pde"].astype(float)

training_loss_log = np.log10(training_loss).astype(float)
validation_loss_log = np.log10(validation_loss).astype(float)
pde_loss_log = np.log10(pde_loss).astype(float)

best_epoch = np.argmin(validation_loss)

best_train = training_loss[best_epoch]
best_val = validation_loss[best_epoch]
best_pde_mse = pde_loss[best_epoch]
best_pde_rms = np.sqrt(best_pde_mse)

#representative numbers
print("Best epoch:", best_epoch + 1)
print("Best val_total:", best_val)
print("Train_total at best epoch:", best_train)
print("PDE RMS at best epoch:", best_pde_rms)

#Log-scale loss curves
plt.figure(figsize=(10,5))
plt.grid(True, alpha=0.3)
plt.plot(training_loss_log, label="Training Loss")
plt.plot(validation_loss_log, label="Validation Loss")
plt.plot(pde_loss_log, label="PDE Residual Loss")
plt.axvline(best_epoch, linestyle="--", alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Loss (log10)")
plt.legend(frameon=False)
plt.title("Training and Validation Loss Over Time")
plt.grid(True)
plt.show()

#PDE metrics evaluation (used in stage 3)
pde_mse = pde_loss
pde_rms = np.sqrt(np.maximum(pde_mse, 0.0)) 

#extract tail metrics
tail_frac = 0.20                         
tail_n = max(1, int(len(pde_rms) * tail_frac))
tail_slice = slice(len(pde_rms) - tail_n, len(pde_rms))
pde_tail = pde_rms[tail_slice]

pde_bestval_rms = pde_rms[best_epoch]
pde_min_rms = float(np.min(pde_rms))
pde_min_epoch = int(np.argmin(pde_rms))

pde_tail_median = float(np.median(pde_tail))
pde_tail_mean = float(np.mean(pde_tail))
pde_tail_p90 = float(np.quantile(pde_tail, 0.90))

pde_auc = float(np.trapz(pde_rms, dx=1.0)) / max(1, len(pde_rms) - 1)

pde_threshold = 1.25 * pde_bestval_rms   
frac_below = float(np.mean(pde_rms <= pde_threshold))

print("\n===== Unweighted PDE Metrics =====")
print(f"PDE RMS at best-val epoch: {pde_bestval_rms:.6g}  (epoch {best_epoch + 1})")
print(f"Minimum PDE RMS achieved: {pde_min_rms:.6g}  (epoch {pde_min_epoch + 1})")
print(f"Tail PDE RMS (last {int(tail_frac*100)}%) median: {pde_tail_median:.6g}")
print(f"Tail PDE RMS (last {int(tail_frac*100)}%) mean: {pde_tail_mean:.6g}")
print(f"Tail PDE RMS (last {int(tail_frac*100)}%) p90: {pde_tail_p90:.6g}")
print(f"PDE RMS AUC/epoch (avg): {pde_auc:.6g}")
print(f"Frac epochs PDE RMS <= {pde_threshold:.6g}: {frac_below:.3f}")

plt.figure(figsize=(10,5))
plt.grid(True, alpha=0.3)
plt.plot(pde_rms, label="PDE RMS (unweighted)")
plt.axvline(best_epoch, linestyle="--", alpha=0.5, label="Best val epoch")
plt.axvline(pde_min_epoch, linestyle=":", alpha=0.7, label="Min PDE epoch")
plt.axhline(pde_threshold, linestyle="--", alpha=0.3, label="PDE threshold (example)")
plt.axvspan(len(pde_rms) - tail_n, len(pde_rms) - 1, alpha=0.1, label=f"Tail ({int(tail_frac*100)}%)")
plt.xlabel("Epoch")
plt.ylabel("PDE RMS")
plt.title("PDE Metrics Over Training (Unweighted)")
plt.legend(frameon=False)
plt.show()

