#!/bin/bash
# ============================================================
#  run_calibration.sh
#  Lance la calibration Heston + SABR pour SPY, AAPL, NVDA, GLD, TSLA
#  Copie les PNGs générés vers ~/Desktop/
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP="$HOME/Desktop"

echo "============================================================"
echo "  Calibration multi-actifs — Heston + SABR"
echo "  Dossier : $SCRIPT_DIR"
echo "============================================================"

# ── 1. Installer les dépendances Python si besoin ─────────────
echo ""
echo "→ Vérification des dépendances Python..."
pip3 install --quiet yfinance scipy matplotlib pandas numpy

# ── 2. Lancer le script de calibration ───────────────────────
echo ""
echo "→ Lancement de la calibration pour : SPY AAPL NVDA GLD TSLA"
echo ""
cd "$SCRIPT_DIR"
python3 multi_calibration.py SPY AAPL NVDA GLD TSLA

# ── 3. Copier les PNGs vers le Desktop ────────────────────────
echo ""
echo "→ Copie des PNGs vers $DESKTOP ..."
for png in calibration_SPY.png calibration_AAPL.png calibration_NVDA.png calibration_GLD.png calibration_TSLA.png; do
    if [ -f "$SCRIPT_DIR/$png" ]; then
        cp "$SCRIPT_DIR/$png" "$DESKTOP/$png"
        echo "  ✓ $png"
    else
        echo "  ✗ $png introuvable"
    fi
done

echo ""
echo "============================================================"
echo "  Terminé ! PNGs disponibles sur le Bureau."
echo "============================================================"
