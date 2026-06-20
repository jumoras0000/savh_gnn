"""
Script de visualisation des résultats Phase 1.
Extrait l'historique d'entraînement et crée des graphiques d'analyse.
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/phase1/sovereign_encoder_v1.pth"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("VISUALISATION PHASE 1 - PRÉ-ENTRAÎNEMENT MGM")
print("=" * 70)

# Charger le checkpoint
try:
    print(f"\n📂 Chargement du checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    print("✓ Checkpoint chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    exit(1)

# Extraire les informations
history = ckpt.get("history", [])
config = ckpt.get("config", {})
final_epoch = ckpt.get("epoch", "??")
best_val_loss = ckpt.get("best_val_loss")
train_loss = ckpt.get("train_loss")
val_loss = ckpt.get("val_loss")


def _fmt(v):
    """Formate en .6f si numérique, sinon '??' (checkpoint incomplet)."""
    return f"{v:.6f}" if isinstance(v, (int, float)) else "??"


print("\n📊 Informations du modèle:")
print(f"   • Époque finale: {final_epoch}")
print(f"   • Meilleure val_loss: {_fmt(best_val_loss)}")
print(f"   • Train loss final: {_fmt(train_loss)}")
print(f"   • Val loss final: {_fmt(val_loss)}")
print(f"   • Architecture: {config}")

if not history:
    print("\n⚠️  Pas d'historique trouvé dans le checkpoint")
    exit(1)

print(f"\n✓ Historique d'entraînement avec {len(history)} époque(s)")

# Extraire les données
epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]
lrs = [h.get("lr", 0) for h in history]

# === GRAPHIQUE 1: LOSS TRAINING vs VALIDATION ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, train_losses, marker='o', label='Train Loss', linewidth=2, markersize=4)
ax.plot(epochs, val_losses, marker='s', label='Val Loss', linewidth=2, markersize=4)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
ax.set_title('Training Curves - Phase 1 MGM Pretraining', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_yscale('log')
plot_path1 = PLOT_DIR / "phase1_training_curves.png"
plt.tight_layout()
plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
print(f"\n📈 Graphique 1 sauvegardé: {plot_path1}")
plt.close()

# === GRAPHIQUE 2: LEARNING RATE SCHEDULE ===
if len(set(lrs)) > 1:  # S'il y a une variation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, lrs, marker='o', color='green', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule - Phase 1', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plot_path2 = PLOT_DIR / "phase1_lr_schedule.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"📈 Graphique 2 sauvegardé: {plot_path2}")
    plt.close()

# === GRAPHIQUE 3: ZOOM SUR LES DERNIÈRES ÉPIQUES (si > 20) ===
if len(epochs) > 20:
    fig, ax = plt.subplots(figsize=(12, 6))
    subset_start = len(epochs) - 20
    ax.plot(epochs[subset_start:], train_losses[subset_start:], marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs[subset_start:], val_losses[subset_start:], marker='s', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
    ax.set_title('Last 20 Epochs - Convergence Detail', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plot_path3 = PLOT_DIR / "phase1_last_epochs.png"
    plt.tight_layout()
    plt.savefig(plot_path3, dpi=150, bbox_inches='tight')
    print(f"📈 Graphique 3 sauvegardé: {plot_path3}")
    plt.close()

# === STATISTIQUES ===
min_val_loss = min(val_losses)
min_val_epoch = epochs[val_losses.index(min_val_loss)]
convergence_improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100)
generalization_gap = val_losses[-1] - train_losses[-1]

print("\n📊 STATISTIQUES D'ENTRAÎNEMENT:")
print(f"   • Loss initiale (train): {train_losses[0]:.6f}")
print(f"   • Loss finale (train): {train_losses[-1]:.6f}")
print(f"   • Amélioration: {convergence_improvement:.2f}%")
print(f"   • Meilleure val_loss: {min_val_loss:.6f} (époque {min_val_epoch})")
print(f"   • Écart généralisation: {generalization_gap:.6f}")

# === ANALYSE DE QUALITÉ ===
print("\n🎯 ANALYSE DE QUALITÉ:")
if convergence_improvement > 30:
    print(f"   ✓ Bon apprentissage ({convergence_improvement:.1f}% d'amélioration)")
else:
    print(f"   ⚠️  Apprentissage modéré ({convergence_improvement:.1f}% d'amélioration)")

if abs(generalization_gap) < 0.001:
    print("   ✓ Excellente généralisation (écart ~0)")
elif generalization_gap < 0.01:
    print(f"   ✓ Bonne généralisation (écart = {generalization_gap:.5f})")
else:
    print(f"   ⚠️  Overfitting détecté (écart = {generalization_gap:.5f})")

if len(history) >= 100:
    print("   ✓ Entraînement complet (100+ épisodes)")
elif len(history) >= 50:
    print(f"   ✓ Entraînement substantiel ({len(history)} épisodes)")
else:
    print(f"   ⚠️  Entraînement court ({len(history)} épisodes)")

print("\n" + "=" * 70)
print("✓ Visualisation terminée!")
print("=" * 70)
