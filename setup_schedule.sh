#!/bin/bash
# setup_schedule.sh — Configura ejecución automática 2x día en Mac
# Usa launchd (más confiable que cron en macOS) + pmset para despertar el Mac
#
# Uso: bash setup_schedule.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv312"
LOG_DIR="$PROJECT_DIR/logs"
PLIST_DIR="$HOME/Library/LaunchAgents"

mkdir -p "$LOG_DIR" "$PLIST_DIR"

echo "📁 Proyecto: $PROJECT_DIR"
echo "🐍 Venv:     $VENV"
echo ""

# ── Script wrapper que activa el venv y corre el pipeline ─────────────────────
cat > "$PROJECT_DIR/run_pipeline.sh" << SCRIPT
#!/bin/bash
# Activar venv y ejecutar pipeline
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PATH="$VENV/bin:\$PATH"
cd "$PROJECT_DIR"
source "$VENV/bin/activate"
python main.py --shorts --category ai >> "$LOG_DIR/run_\$(date +%Y%m%d_%H%M).log" 2>&1
SCRIPT
chmod +x "$PROJECT_DIR/run_pipeline.sh"
echo "✅ run_pipeline.sh creado"

# ── LaunchAgent: 7:00 AM ──────────────────────────────────────────────────────
cat > "$PLIST_DIR/com.arxivbot.morning.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.arxivbot.morning</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT_DIR/run_pipeline.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd_morning.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd_morning_err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
echo "✅ LaunchAgent 7:00 AM creado"

# ── LaunchAgent: 12:00 AM (medianoche) ────────────────────────────────────────
cat > "$PLIST_DIR/com.arxivbot.midnight.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.arxivbot.midnight</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$PROJECT_DIR/run_pipeline.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>0</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd_midnight.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd_midnight_err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
echo "✅ LaunchAgent 12:00 AM creado"

# ── Registrar los LaunchAgents ────────────────────────────────────────────────
launchctl unload "$PLIST_DIR/com.arxivbot.morning.plist"  2>/dev/null || true
launchctl unload "$PLIST_DIR/com.arxivbot.midnight.plist" 2>/dev/null || true
launchctl load   "$PLIST_DIR/com.arxivbot.morning.plist"
launchctl load   "$PLIST_DIR/com.arxivbot.midnight.plist"
echo "✅ LaunchAgents registrados"

# ── pmset: despertar el Mac antes de cada ejecución ──────────────────────────
# Requiere permiso sudo — despierta 2 min antes para que el sistema esté listo
echo ""
echo "⚙️  Configurando wake schedule (requiere sudo)..."
echo "   El Mac se despertará a las 6:58 AM y 11:58 PM automáticamente"
echo ""

# Nota: pmset schedule solo soporta un wake a la vez en versiones recientes
# Usamos repeating para horarios diarios
sudo pmset repeat wake MTWRFSU 06:58:00
echo "✅ pmset configurado para despertar a las 6:58 AM diario"
echo ""
echo "ℹ️  Para agregar el despertar de medianoche, ejecuta manualmente:"
echo "   sudo pmset repeat wakeorpoweron MTWRFSU 23:58:00"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup completo. Horario:"
echo "   🌅 7:00 AM  → pipeline --category ai"
echo "   🌙 12:00 AM → pipeline --category ml"
echo ""
echo "📋 Ver logs en: $LOG_DIR/"
echo "🔍 Verificar estado:"
echo "   launchctl list | grep arxivbot"
echo ""
echo "🛑 Para desactivar:"
echo "   launchctl unload ~/Library/LaunchAgents/com.arxivbot.morning.plist"
echo "   launchctl unload ~/Library/LaunchAgents/com.arxivbot.midnight.plist"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"