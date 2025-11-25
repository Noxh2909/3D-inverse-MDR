import os
from datetime import datetime

start_time = None
console_box = None

def set_console_box(box):
    """Externes UI-Element registrieren, damit Logs ins Fenster geschrieben werden."""
    global console_box
    console_box = box
    log_to_console("Console connected.")

def log_to_console(text):
    """Schreibt Text in die GUI-Console (oder in stdout, falls keine vorhanden)."""
    global console_box
    timestamp = datetime.now().strftime('%H:%M:%S')
    msg = f"[{timestamp}] {text}"
    if console_box is not None:
        try:
            console_box.appendPlainText(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

def write_log_to_file(log_path: str, text: str):
    """Schreibt eine Zeile Text in die angegebene Log-Datei."""
    try:
        with open(log_path, 'a', encoding='utf-8', newline='') as f:
            f.write(text + '\n')
    except Exception as e:
        log_to_console(f"Failed to write log: {e}")
    
def log_session_event(event: str):
    """Loggt ein Ereignis mit Zeitstempel und verstrichener Zeit seit Sitzungsbeginn in eine CSV-Datei."""
    global start_time
    if start_time is None:
        start_time = datetime.now()  
    elapsed = (datetime.now() - start_time).total_seconds()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    base = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"session_log_{start_time.strftime('%Y%m%d_%H%M%S')}.csv")
    write_log_to_file(log_path, f"{timestamp},{elapsed:.2f},{event}")
    log_to_console(event)