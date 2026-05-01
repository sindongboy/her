"""Daemon package — background launchd service for the her assistant.

Phase 3: macOS background daemon with pidfile lifecycle, log rotation,
and wake-word loop via VoiceChannel.

Entry point: python -m apps.daemon
CLI wrapper: bin/her {start|stop|status|logs|restart|install|uninstall}
"""
