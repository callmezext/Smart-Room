import os
import time
import json
import uuid
import copy
import re
import datetime
import shutil
import threading
import subprocess
import asyncio
from typing import Any, List, Optional
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
import cv2

# Initialize FastAPI app
app = FastAPI(title="Smart Room Unified Dashboard")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MUSIC_DIR = os.path.join(BASE_DIR, "music")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "static", "snapshots")
VIDEOS_DIR = os.path.join(BASE_DIR, "static", "videos")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Mount Static Directories
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# JSON Database Files
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
PREFERENCES_FILE = os.path.join(DATA_DIR, "preferences.json")
ALARMS_FILE = os.path.join(DATA_DIR, "alarms.json")
LOGS_FILE = os.path.join(DATA_DIR, "logs.json")
TODO_FILE = os.path.join(DATA_DIR, "todos.json")
ROUTINES_FILE = os.path.join(DATA_DIR, "routines.json")
FAVORITES_FILE = os.path.join(DATA_DIR, "favorites.json")
REMINDERS_FILE = os.path.join(DATA_DIR, "reminders.json")
USER_PROFILE_FILE = os.path.join(DATA_DIR, "user_profile.json")
BLUETOOTH_FILE = os.path.join(DATA_DIR, "bluetooth.json")

# Global state and locks
state_lock = threading.Lock()
audio_lock = asyncio.Lock()

# Audio playback states
active_play_process = None
active_music_process = None
current_playing_name = None
PLAYED_MUSIC_HISTORY = []
is_alarm_playing = False
active_alarm_id = None
active_alarm_name = None
active_alarm_type = None
active_alarm_value = None
active_alarm_volume = 100
pre_alarm_volume = None  # To restore if user wants, but default_volume takes priority

# Snoozed alarms: alarm_id -> trigger_epoch
snoozed_alarms = {}

# CCTV States
camera_lock = threading.Lock()
cap = None
camera_active = False
latest_frame = None
is_recording_video = False
active_camera_clients = 0

# Presence & Motion Detection States
user_presence_state = "unknown"  # "present", "absent", "unknown"
motion_detected_while_away = False
motion_detected_timestamp = None
last_camera_motion_time = time.time()



# Bluetooth States
bluetooth_scanning = False
bluetooth_scan_results = []

# Chat History
CHAT_HISTORY = []

# Constants & Defaults
DAYS_ID = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
DAY_MAP = {
    "senin": "monday",
    "selasa": "tuesday",
    "rabu": "wednesday",
    "kamis": "thursday",
    "jumat": "friday",
    "sabtu": "saturday",
    "minggu": "sunday",
}
DAY_LABELS = {
    "monday": "Senin",
    "tuesday": "Selasa",
    "wednesday": "Rabu",
    "thursday": "Kamis",
    "friday": "Jumat",
    "saturday": "Sabtu",
    "sunday": "Minggu",
}

DEFAULT_SYSTEM_PROMPT = (
    "Anda adalah Zex, asisten suara pintar berbasis Gemini AI untuk kamar tidur pintar (Smart Room). "
    "Anda dibekali dengan berbagai tools canggih (alarm, cuaca, pencarian web, pemutar musik lokal, radio streaming, YouTube, timer, mode rutinitas, shortcut favorit, briefing, dan pengatur volume).\n\n"
    "PRINSIP KOMUNIKASI (SANGAT PENTING):\n"
    "1. RESPONS SINGKAT & PADAT: Karena respons Anda akan diubah menjadi suara (TTS) di speaker, jawablah dengan sangat singkat, ramah, langsung ke inti, dan MAKSIMAL 2 KALIMAT.\n"
    "2. GAYA BAHASA: Pakai Bahasa Indonesia yang santai, ringan, natural, tidak kaku, dan tidak terdengar seperti AI. Diperbolehkan memanggil pengguna dengan sebutan 'tuan' dan menggunakan kata 'baik' untuk mengonfirmasi perintah. Jangan menyebut diri sendiri dengan nama 'Zex' saat berbicara, gunakan kata 'saya' untuk menunjuk diri Anda sendiri. Jangan pakai kata 'aku'.\n"
    "3. MINIM KATA MAAF: Jangan meminta maaf berulang-ulang. Jika ada kesalahan, cukup singkat dan santai.\n"
    "5. AUTO-CORRECT KESALAHAN SUARA (VOICE-TO-TEXT): Karena input teks didapat dari transkripsi suara, seringkali terjadi kesalahan kata (typo) akibat pelafalan (misal: 'putar musik pop' didengar 'putar musik mpop', 'sheila on 7' didengar 'sila on seven', 'nyalakan kamera' didengar 'nyala akamera'). "
    "Anda harus secara cerdas menerjemahkan maksud pengguna. Jika pengguna ingin memutar musik lokal yang mirip namanya dengan daftar lagu lokal yang tersedia, panggilah play_local_music dengan nama file yang benar.\n\n"
    "PANDUAN PENGGUNAAN TOOLS:\n"
    "1. ALARM (set_alarm): gunakan untuk alarm jam tertentu.\n"
    "2. TIMER (set_timer): gunakan untuk countdown seperti '10 menit lagi'.\n"
    "3. REMINDER BERULANG (set_recurring_reminder): gunakan untuk pengingat harian/mingguan.\n"
    "4. MUSIK: gunakan play_music untuk radio genre, play_local_music untuk file lokal (pilih versi berdasarkan durasi di context, e.g., durasi panjang 2:52 untuk 'full' atau pendek 0:32 untuk '30 detik'/'pendek'), play_youtube untuk lagu dari YouTube, dan stop_music untuk menghentikan. Jika user meminta 'sebelumnya', 'putar lagunya lagi', atau 'putar musik sebelumnya', panggil tool play_previous_music.\n"
    "5. VOLUME (set_volume): gunakan untuk set volume 0 sampai 100.\n"
    "6. BRIEFING (get_briefing): gunakan untuk ringkasan pagi/harian.\n"
    "7. MEMORI (remember_fact): gunakan untuk menyimpan preferensi, jadwal, nama, kota default, dan kebiasaan user.\n"
    "8. TO-DO (add_todo / complete_todo): gunakan untuk mencatat dan menuntaskan tugas.\n"
    "9. ROUTINE/FAVORITE: gunakan run_routine and run_favorite untuk preset yang sudah tersedia."
)

DEFAULT_SETTINGS = {
    "api_keys": [],
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "volume": 80,
    "default_volume": 50,  # Smart volume default
    "active_model": "gemini-3.5-flash",
    "stt_lang": "id-ID",
    "tts_voice": "female",
    "response_mode": "normal",
    "cctv_snapshot_enabled": False,
    "cctv_snapshot_interval": 10,  # minutes
    "alarm_briefing_enabled": True,
    "keepalive_interval_min": 5,
    "camera_auto_off": True,
    "immich_api_key": "",
    "immich_address": "http://127.0.0.1:2283",
    "immich_sync_enabled": False,
    "wifi_sensing_enabled": False,
    "wifi_sensing_method": "ping",
    "wifi_sensing_target_ip": "",
    "wifi_sensing_target_mac": "",
    "cctv_motion_detection_enabled": False,
    "presence_stillness_threshold_min": 3.0,
    "presence_short_limit_min": 15.0,
    "presence_medium_limit_min": 60.0,
    "presence_long_limit_min": 120.0,
    "cctv_fps_day": 24,
    "cctv_fps_night": 10,
    "cctv_fps_night_start_hour": 0,
    "cctv_fps_night_end_hour": 4,
    "tts_engine": "edge-tts",
    "stt_engine": "browser",
}






DEFAULT_PREFERENCES = {
    "response_mode": "normal",
    "default_city": "Malang",
    "briefing_time": "07:00",
    "wake_time": "06:00",
    "quiet_hours_enabled": True,
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "06:00",
    "favorite_genre": "lofi",
    "memories": {},
}

DEFAULT_ROUTINES = {
    "mode tidur": {
        "name": "mode tidur",
        "description": "Volume kecil. Musik cuma kalau diminta.",
        "actions": [
            {"type": "set_volume", "level": 20},
        ],
    },
    "mode kerja": {
        "name": "mode kerja",
        "description": "Volume sedang dan musik fokus.",
        "actions": [
            {"type": "set_volume", "level": 40},
            {"type": "play_music", "genre": "lofi"},
        ],
    },
    "mode santai": {
        "name": "mode santai",
        "description": "Musik chill dan volume nyaman.",
        "actions": [
            {"type": "set_volume", "level": 35},
            {"type": "play_music", "genre": "jazz"},
        ],
    },
}

DEFAULT_FAVORITES = {
    "radio santai": {
        "name": "radio santai",
        "type": "play_music",
        "payload": {"genre": "relax"},
    },
    "radio fokus": {
        "name": "radio fokus",
        "type": "play_music",
        "payload": {"genre": "lofi"},
    },
    "radio pop": {
        "name": "radio pop",
        "type": "play_music",
        "payload": {"genre": "pop"},
    },
}

DEFAULT_BLUETOOTH_SETTINGS = {
    "auto_reconnect_mac": None,
    "auto_reconnect_enabled": False,
    "auto_switch_to_bt": True   # Auto set BT as default sink when connected
}

ALLOWED_MUSIC_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
MUSIC_STREAMS = {
    "lofi": "http://stream.laut.fm/lofi",
    "relax": "http://stream.laut.fm/chillout",
    "jazz": "http://jazzblues.ice.infomaniak.ch/jazzblues-high.mp3",
    "pop": "https://novazz.ice.infomaniak.ch/novazz-128.mp3",
}

# --- HELPER DATABASE FUNCTIONS ---

def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)

def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return deep_copy(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return deep_copy(default)
            return json.loads(raw)
    except Exception:
        return deep_copy(default)

def write_json_file(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing json file {path}: {e}")

def ensure_file(path: str, default: Any):
    if not os.path.exists(path):
        write_json_file(path, default)

def ensure_all_files():
    ensure_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    ensure_file(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    ensure_file(ROUTINES_FILE, DEFAULT_ROUTINES)
    ensure_file(FAVORITES_FILE, DEFAULT_FAVORITES)
    ensure_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    
    if not os.path.exists(ALARMS_FILE) or os.path.getsize(ALARMS_FILE) == 0:
        write_json_file(ALARMS_FILE, [])
    if not os.path.exists(LOGS_FILE) or os.path.getsize(LOGS_FILE) == 0:
        write_json_file(LOGS_FILE, [])
    if not os.path.exists(TODO_FILE) or os.path.getsize(TODO_FILE) == 0:
        write_json_file(TODO_FILE, [])
    if not os.path.exists(REMINDERS_FILE) or os.path.getsize(REMINDERS_FILE) == 0:
        write_json_file(REMINDERS_FILE, [])
    if not os.path.exists(USER_PROFILE_FILE) or os.path.getsize(USER_PROFILE_FILE) == 0:
        write_json_file(USER_PROFILE_FILE, {})

ensure_all_files()

# --- HELPER SYSTEM VOLUME FUNCTIONS ---

def set_system_volume(level: int):
    level = max(0, min(100, int(level)))
    # Set PulseAudio volume for default sink
    if os.environ.get("PULSE_SERVER") or os.path.exists("/tmp/pulse-socket"):
        try:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"], capture_output=True)
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], capture_output=True)
            # Also set volume specifically for active Bluetooth speaker if connected
            bt_sink = get_active_bt_sink()
            if bt_sink:
                subprocess.run(["pactl", "set-sink-volume", bt_sink, f"{level}%"], capture_output=True)
                subprocess.run(["pactl", "set-sink-mute", bt_sink, "0"], capture_output=True)
        except Exception:
            pass
    # Set ALSA Master fallback
    try:
        subprocess.run(["amixer", "-c", "0", "sset", "Master", f"{level}%", "unmute"], capture_output=True)
        subprocess.run(["amixer", "-c", "0", "sset", "Speaker", f"{level}%", "unmute"], capture_output=True)
        subprocess.run(["amixer", "-c", "0", "sset", "Headphone", f"{level}%", "unmute"], capture_output=True)
    except Exception:
        pass

def get_system_volume() -> int:
    if os.environ.get("PULSE_SERVER") or os.path.exists("/tmp/pulse-socket"):
        try:
            res = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], capture_output=True, text=True)
            if res.returncode == 0:
                matches = re.findall(r'(\d+)%', res.stdout)
                if matches:
                    return int(matches[0])
        except Exception:
            pass
    try:
        res = subprocess.run(["amixer", "-c", "0", "sget", "Master"], capture_output=True, text=True)
        if res.returncode == 0:
            matches = re.findall(r'\[(\d+)%\]', res.stdout)
            if matches:
                return int(matches[0])
    except Exception:
        pass
    return 50

# --- SMART RESTORATION VOLUME LOGIC ---

def get_configured_default_volume() -> int:
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    return int(settings.get("default_volume", 50))

# Set initial volume on startup to default volume
set_system_volume(get_configured_default_volume())

# --- AUDIO LOGGING ---

def add_log(event_type: str, alarm_name: str, message: str):
    with state_lock:
        logs = read_json_file(LOGS_FILE, [])
        logs.append({
            "timestamp": int(time.time()),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "event_type": event_type,
            "alarm_name": alarm_name,
            "message": message
        })
        logs = logs[-100:]  # Keep only last 100 logs
        write_json_file(LOGS_FILE, logs)

# --- ALARMS AND KEEPALIVE LOGIC ---

def get_alarms() -> list:
    return read_json_file(ALARMS_FILE, [])

def save_alarms(alarms: list):
    write_json_file(ALARMS_FILE, alarms)

def stop_active_audio():
    global active_play_process, is_alarm_playing, active_alarm_id, active_alarm_name, active_alarm_type, active_alarm_value
    was_alarm_playing = False
    with state_lock:
        was_alarm_playing = is_alarm_playing
        if active_play_process is not None:
            try:
                active_play_process.terminate()
                active_play_process.wait(timeout=0.5)
            except Exception:
                try:
                    active_play_process.kill()
                except Exception:
                    pass
            active_play_process = None
        is_alarm_playing = False
        active_alarm_id = None
        active_alarm_name = None
        active_alarm_type = None
        active_alarm_value = None
    
    # Smart volume restoration: only set volume back to DEFAULT volume if an alarm was actually ringing
    if was_alarm_playing:
        default_vol = get_configured_default_volume()
        set_system_volume(default_vol)
        add_log("VOLUME_RESTORE", "System", f"Volume otomatis dikembalikan ke default: {default_vol}%")

def get_active_bt_sink() -> Optional[str]:
    """Returns the PipeWire/PulseAudio Bluetooth (bluez) sink name, or None if not found."""
    try:
        sink_res = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True)
        if sink_res.returncode == 0:
            for line in sink_res.stdout.splitlines():
                if "bluez" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return None

def play_keepalive_audio():

    global is_alarm_playing
    with state_lock:
        if is_alarm_playing:
            return
            
    keepalive_path = os.path.join(UPLOAD_DIR, "keepalive.wav")
    if not os.path.exists(keepalive_path):
        try:
            # Generate a 2.0s 20Hz sub-bass wave at low volume
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=20:duration=2.0", "-af", "volume=0.01", keepalive_path],
                capture_output=True, text=True
            )
        except Exception as e:
            print(f"Error generating keepalive sound: {e}")
            return
            
    try:
        bt_sink = get_active_bt_sink()
        if os.environ.get("PULSE_SERVER") or os.path.exists("/tmp/pulse-socket"):
            if bt_sink:
                cmd = ["paplay", "--device", bt_sink, keepalive_path]
            else:
                cmd = ["paplay", keepalive_path]
        else:
            cmd = ["aplay", "-q", keepalive_path]
            
        subprocess.run(cmd, capture_output=True)
        add_log("KEEPALIVE", "Speaker Keepalive", "Memutar suara pancingan agar speaker bluetooth tidak mati")
    except Exception as e:
        print(f"Error playing keepalive sound: {e}")


def play_audio_file(file_path: str, volume: Optional[int] = None, is_alarm: bool = False):
    global active_play_process, is_alarm_playing
    
    # If not playing an alarm, stop active first
    if not is_alarm:
        stop_active_audio()
    else:
        # If it is an alarm, stop whatever is playing first
        with state_lock:
            if active_play_process is not None:
                try:
                    active_play_process.terminate()
                except Exception:
                    pass
    
    if volume is None:
        settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
        volume = settings.get("volume", get_configured_default_volume())
        
    # Set volume to target (e.g. current volume or alarm volume)
    set_system_volume(volume)
    
    # Ensure WAV format for playing
    wav_path = file_path
    if not file_path.lower().endswith(".wav"):
        wav_path = file_path.rsplit(".", 1)[0] + "_play.wav"
        if not os.path.exists(wav_path):
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", wav_path],
                    capture_output=True, text=True
                )
            except Exception as e:
                print(f"Conversion error: {e}")
                wav_path = file_path # Fallback
                
    # Determine command - auto-select BT sink if connected
    cmd = None
    bt_sink = get_active_bt_sink()
    if os.environ.get("PULSE_SERVER") or os.path.exists("/tmp/pulse-socket"):
        if bt_sink:
            cmd = ["paplay", "--device", bt_sink, wav_path]
        else:
            cmd = ["paplay", wav_path]
    else:
        cmd = ["aplay", "-q", wav_path]
        
    try:
        with state_lock:
            active_play_process = subprocess.Popen(cmd)
            is_alarm_playing = is_alarm
        return True
    except Exception as e:
        print(f"Error spawning player process: {e}")
        is_alarm_playing = False
        return False

# --- TTS GENERATION & PLAYBACK ---

VOICE_MAPPING = {
    "id": "id-ID-GadisNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "id-ID-GadisNeural": "id-ID-GadisNeural",
    "id-ID-ArdiNeural": "id-ID-ArdiNeural",
    "en-US-JennyNeural": "en-US-JennyNeural",
    "en-US-GuyNeural": "en-US-GuyNeural",
    "ja-JP-NanamiNeural": "ja-JP-NanamiNeural",
    "ja-JP-KeitaNeural": "ja-JP-KeitaNeural",
    "ko-KR-SunHiNeural": "ko-KR-SunHiNeural",
    "ko-KR-InJoonNeural": "ko-KR-InJoonNeural"
}

async def generate_tts_file_async(text: str, lang: str, output_path: str):
    if lang == "jw" or lang not in VOICE_MAPPING:
        loop = asyncio.get_running_loop()
        def _save_gtts():
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang=lang)
                tts.save(output_path)
            except Exception as e:
                print(f"gTTS generation failed: {e}")
        await loop.run_in_executor(None, _save_gtts)
    else:
        voice = VOICE_MAPPING[lang]
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        except Exception as e:
            print(f"Edge TTS failed: {e}. Falling back to gTTS.")
            loop = asyncio.get_running_loop()
            def _save_fallback():
                try:
                    from gtts import gTTS
                    fallback_lang = "id" if lang not in ["id", "en", "ja", "ko"] else lang
                    tts = gTTS(text=text, lang=fallback_lang)
                    tts.save(output_path)
                except Exception as ex:
                    print(f"Fallback gTTS failed: {ex}")
            await loop.run_in_executor(None, _save_fallback)

def generate_tts_file_sync(text: str, lang: str, output_path: str):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # Fallback to gtts synchronously inside run_in_executor
        from gtts import gTTS
        fallback_lang = "id" if lang not in ["id", "en", "ja", "ko", "jw"] else lang
        tts = gTTS(text=text, lang=fallback_lang)
        tts.save(output_path)
    else:
        try:
            loop.run_until_complete(generate_tts_file_async(text, lang, output_path))
        except Exception as e:
            print(f"Sync TTS failed: {e}")

# Formatting TTS Strings
def get_greeting():
    hour = time.localtime().tm_hour
    if 5 <= hour < 11:
        return "Selamat Pagi"
    elif 11 <= hour < 15:
        return "Selamat Siang"
    elif 15 <= hour < 18:
        return "Selamat Sore"
    else:
        return "Selamat Malam"

def get_day_name():
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    return days[time.localtime().tm_wday]

def format_tts_text(text: str, alarm_time: str) -> str:
    formatted = text
    formatted = formatted.replace("{greeting}", get_greeting())
    formatted = formatted.replace("{salam}", get_greeting())
    formatted = formatted.replace("{time}", alarm_time)
    formatted = formatted.replace("{waktu}", alarm_time)
    formatted = formatted.replace("{day}", get_day_name())
    formatted = formatted.replace("{hari}", get_day_name())
    return formatted

def trigger_alarm(alarm_id: str, alarm_name: str, alarm_type: str, alarm_value: str, volume: int, tts_lang: str = "id"):
    global active_alarm_id, active_alarm_name, active_alarm_type, active_alarm_value, active_alarm_volume, is_alarm_playing
    
    active_alarm_id = alarm_id
    active_alarm_name = alarm_name
    active_alarm_type = alarm_type
    active_alarm_volume = volume
    
    add_log("TRIGGER", alarm_name, f"Alarm berdering ({alarm_type.upper()}) pada volume {volume}%")
    
    if alarm_type == "tts":
        alarm_time = time.strftime("%H:%M", time.localtime())
        formatted_text = format_tts_text(alarm_value, alarm_time)
        active_alarm_value = formatted_text
        
        tts_filename = f"tts_{alarm_id}.mp3"
        tts_path = os.path.join(UPLOAD_DIR, tts_filename)
        
        try:
            generate_tts_file_sync(formatted_text, tts_lang, tts_path)
            play_audio_file(tts_path, volume, is_alarm=True)
        except Exception as e:
            print(f"Error triggering TTS Alarm: {e}")
            add_log("ERROR", alarm_name, f"Gagal membuat suara alarm: {str(e)}")
            is_alarm_playing = False
    else:
        active_alarm_value = alarm_value
        file_path = os.path.join(UPLOAD_DIR, alarm_value)
        if os.path.exists(file_path):
            play_audio_file(file_path, volume, is_alarm=True)
        else:
            add_log("ERROR", alarm_name, f"File ringtone '{alarm_value}' tidak ditemukan")
            is_alarm_playing = False

# --- BACKGROUND MONITOR LOOP ---

def alarm_monitor_loop():
    global snoozed_alarms
    print("Background alarm monitor loop started...")
    keepalive_played_this_minute = False
    
    while True:
        try:
            now = time.localtime()
            current_time_str = time.strftime("%H:%M", now)
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            current_day_str = day_names[now.tm_wday]
            current_epoch = time.time()
            
            # Keep bluetooth speaker awake - interval from settings
            ka_settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
            ka_interval = int(ka_settings.get("keepalive_interval_min", 5))
            if ka_interval <= 0:
                # 0 means disabled
                keepalive_played_this_minute = False
            elif now.tm_min % ka_interval == 0:
                if not keepalive_played_this_minute:
                    keepalive_played_this_minute = True
                    threading.Thread(target=play_keepalive_audio, daemon=True).start()
            else:
                keepalive_played_this_minute = False
                
            # 1. Check snoozed alarms
            snoozed_to_trigger = []
            for alarm_id, trigger_epoch in list(snoozed_alarms.items()):
                if current_epoch >= trigger_epoch:
                    snoozed_to_trigger.append(alarm_id)
                    
            for alarm_id in snoozed_to_trigger:
                alarms = get_alarms()
                alarm = next((a for a in alarms if a["id"] == alarm_id), None)
                if alarm:
                    trigger_alarm(
                        alarm_id=alarm["id"],
                        alarm_name=alarm["name"],
                        alarm_type=alarm["type"],
                        alarm_value=alarm["tts_text"] if alarm["type"] == "tts" else alarm["ringtone"],
                        volume=alarm["volume"],
                        tts_lang=alarm.get("tts_lang", "id")
                    )
                snoozed_alarms.pop(alarm_id, None)
                
            # 2. Check standard alarms
            alarms = get_alarms()
            updated_alarms = False
            
            for alarm in alarms:
                if alarm["enabled"]:
                    if alarm["time"] == current_time_str:
                        last_triggered = alarm.get("last_triggered", "")
                        today_minute_str = time.strftime("%Y-%m-%d %H:%M", now)
                        
                        if last_triggered != today_minute_str:
                            repeat = alarm.get("repeat_days", [])
                            is_today_trigger = False
                            
                            if not repeat:
                                is_today_trigger = True
                                alarm["enabled"] = False # Disable one-shot alarm
                                updated_alarms = True
                            elif current_day_str in repeat:
                                is_today_trigger = True
                                
                            if is_today_trigger:
                                alarm["last_triggered"] = today_minute_str
                                updated_alarms = True
                                
                                trigger_alarm(
                                    alarm_id=alarm["id"],
                                    alarm_name=alarm["name"],
                                    alarm_type=alarm["type"],
                                    alarm_value=alarm["tts_text"] if alarm["type"] == "tts" else alarm["ringtone"],
                                    volume=alarm["volume"],
                                    tts_lang=alarm.get("tts_lang", "id")
                                )
                                break # trigger one alarm per tick
                                
            if updated_alarms:
                save_alarms(alarms)
                
        except Exception as e:
            print(f"Error in alarm monitor thread: {e}")
            
        time.sleep(10)

# monitor_thread = threading.Thread(target=alarm_monitor_loop, daemon=True)
# monitor_thread.start()

# --- BLUETOOTH HELPER FUNCTIONS & DAEMON ---

def get_bluetooth_controller_status():
    res = subprocess.run(["bluetoothctl", "show"], capture_output=True, text=True)
    powered = False
    discovering = False
    name = "Bluetooth Adapter"
    if res.returncode == 0:
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("Powered:"):
                powered = "yes" in line_str.lower()
            elif line_str.startswith("Discovering:"):
                discovering = "yes" in line_str.lower()
            elif line_str.startswith("Name:"):
                name = line_str.split(":", 1)[1].strip()
    return {"powered": powered, "discovering": discovering, "name": name}

def get_bluetooth_devices():
    paired_macs = set()
    connected_macs = set()
    all_devices = []

    # Get paired
    res_paired = subprocess.run(["bluetoothctl", "devices", "Paired"], capture_output=True, text=True)
    if res_paired.returncode == 0:
        for line in res_paired.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) >= 3 and parts[0] == "Device":
                paired_macs.add(parts[1])

    # Get connected
    res_connected = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True)
    if res_connected.returncode == 0:
        for line in res_connected.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) >= 3 and parts[0] == "Device":
                connected_macs.add(parts[1])

    # Get all known
    res_all = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True)
    if res_all.returncode == 0:
        for line in res_all.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) >= 3 and parts[0] == "Device":
                mac = parts[1]
                name = parts[2]
                all_devices.append({
                    "mac": mac,
                    "name": name,
                    "paired": mac in paired_macs,
                    "connected": mac in connected_macs
                })
    return {
        "all": all_devices,
        "paired": [d for d in all_devices if d["paired"]],
        "connected": [d for d in all_devices if d["connected"]]
    }

def run_bluetooth_scan():
    global bluetooth_scanning
    bluetooth_scanning = True
    # Scan for 8 seconds
    subprocess.run(["timeout", "8", "bluetoothctl", "scan", "on"], capture_output=True)
    bluetooth_scanning = False

def fix_bluetooth_audio_routing(mac: str):
    """
    After BT reconnect, ensure PipeWire/PulseAudio routes audio properly:
    1. Wait briefly for sink to register
    2. Set BT sink as default sink (if auto_switch_to_bt enabled)
    3. Move all existing sink-inputs (audio streams) to BT sink
    4. Restore volume to configured default
    """
    bt_cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    if not bt_cfg.get("auto_switch_to_bt", True):
        print(f"BT Audio Fix: auto_switch_to_bt disabled, skipping routing fix")
        return

    # Give PipeWire time to register the new sink
    time.sleep(3)
    
    try:
        formatted_mac = mac.replace(":", "_").upper()
        
        # Find the BT sink name
        sink_res = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True)
        bt_sink = None
        if sink_res.returncode == 0:
            for line in sink_res.stdout.splitlines():
                if "bluez" in line.lower() or formatted_mac in line.upper():
                    parts = line.split()
                    if len(parts) >= 2:
                        bt_sink = parts[1]
                        break
        
        if not bt_sink:
            print(f"BT Audio Fix: No bluez sink found for {mac}, skipping routing fix")
            return
        
        print(f"BT Audio Fix: Setting default sink to {bt_sink}")
        
        # 1. Set as default sink
        subprocess.run(["pactl", "set-default-sink", bt_sink], capture_output=True)
        
        # 2. Move all current sink-inputs to the BT sink
        inputs_res = subprocess.run(["pactl", "list", "sink-inputs", "short"], capture_output=True, text=True)
        if inputs_res.returncode == 0:
            for line in inputs_res.stdout.splitlines():
                if line.strip():
                    input_id = line.split()[0]
                    subprocess.run(["pactl", "move-sink-input", input_id, bt_sink], capture_output=True)
                    print(f"BT Audio Fix: Moved sink-input {input_id} -> {bt_sink}")
        
        # 3. Restore volume on BT sink
        default_vol = get_configured_default_volume()
        subprocess.run(["pactl", "set-sink-volume", bt_sink, f"{default_vol}%"], capture_output=True)
        subprocess.run(["pactl", "set-sink-mute", bt_sink, "0"], capture_output=True)
        
        # 4. Also unmute all sink-inputs
        if inputs_res.returncode == 0:
            for line in inputs_res.stdout.splitlines():
                if line.strip():
                    input_id = line.split()[0]
                    subprocess.run(["pactl", "set-sink-input-mute", input_id, "0"], capture_output=True)
        
        add_log("BLUETOOTH_AUDIO_FIXED", "Bluetooth", f"Routing audio diperbaiki ke {bt_sink} (volume {default_vol}%)")
        print(f"BT Audio Fix: Done. Audio routed to {bt_sink} at {default_vol}%")
        
    except Exception as e:
        print(f"Error in fix_bluetooth_audio_routing: {e}")


def bluetooth_auto_reconnect_loop():
    print("Bluetooth auto-reconnect loop started...")
    consecutive_failures = 0
    while True:
        try:
            bt_cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
            if bt_cfg.get("auto_reconnect_enabled") and bt_cfg.get("auto_reconnect_mac"):
                mac = bt_cfg["auto_reconnect_mac"]
                
                # Check if MAC is connected via bluetoothctl
                res = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True)
                connected = False
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        if mac.lower() in line.lower():
                            connected = True
                            break
                
                # Check if MAC has a working PipeWire sink
                # (to prevent "ghost connection" where it is connected but has no audio device)
                if connected:
                    formatted_mac = mac.replace(":", "_").upper()
                    try:
                        sink_res = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True)
                        if sink_res.returncode == 0:
                            sink_exists = False
                            for line in sink_res.stdout.splitlines():
                                if formatted_mac in line.upper() or "bluez" in line.lower():
                                    sink_exists = True
                                    break
                            if not sink_exists:
                                print(f"Auto-Reconnect: Device {mac} connected but no audio sink found. Ghost connection detected!")
                                add_log("BLUETOOTH_GHOST", "Bluetooth", "Deteksi koneksi hantu (terhubung tapi tidak ada output suara). Memutus koneksi untuk reset...")
                                subprocess.run(["bluetoothctl", "disconnect", mac], capture_output=True)
                                connected = False
                                time.sleep(5)  # Wait before trying to reconnect
                            else:
                                # Sink exists — make sure it's set as default sink
                                # (handles case where system restarted and BT reconnected but PW didn't route properly)
                                default_sink_res = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True)
                                if "bluez" not in default_sink_res.stdout.lower() and formatted_mac not in default_sink_res.stdout.upper():
                                    print(f"Auto-Reconnect: BT sink exists but not default. Fixing routing...")
                                    fix_bluetooth_audio_routing(mac)
                    except Exception as e:
                        print(f"Error checking pulse sink: {e}")
                
                if not connected:
                    print(f"Auto-Reconnect: Device {mac} disconnected. Retrying...")
                    add_log("BLUETOOTH_RECONNECT", "Bluetooth", f"Mencoba auto-connect ke {mac}...")
                    
                    # Connect sequence
                    subprocess.run(["bluetoothctl", "trust", mac], capture_output=True)
                    connect_res = subprocess.run(["bluetoothctl", "connect", mac], capture_output=True, text=True)
                    
                    if "Connection successful" in connect_res.stdout or "successful" in connect_res.stdout.lower():
                        add_log("BLUETOOTH_RECONNECT_SUCCESS", "Bluetooth", f"Berhasil auto-connect ke {mac}")
                        consecutive_failures = 0
                        # Fix audio routing after successful reconnect
                        fix_bluetooth_audio_routing(mac)
                    else:
                        consecutive_failures += 1
                        print(f"Auto-Reconnect: Connection failed. Consecutive failures: {consecutive_failures}")
                        
                        # Avoid power cycling the adapter automatically to prevent interrupting scans/pairing
                        if consecutive_failures >= 3:
                            consecutive_failures = 0
                else:
                    consecutive_failures = 0
        except Exception as e:
            print(f"Error in bluetooth auto-reconnect loop: {e}")
            
        time.sleep(30)


# bt_thread = threading.Thread(target=bluetooth_auto_reconnect_loop, daemon=True)
# bt_thread.start()

def speak_sync(text: str):
    """Synchronous wrapper for speak (async) to call from background thread."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(speak(text))
        loop.close()
    except Exception as e:
        print(f"Error in speak_sync: {e}")

def wifi_sensing_loop():
    """Background loop to check phone presence via Wi-Fi/Ping/Sniffing."""
    print("Wi-Fi sensing presence detection loop started...")
    global user_presence_state, motion_detected_while_away, motion_detected_timestamp
    last_state = "unknown"  # "present", "absent", "unknown"
    last_seen_time = time.time()
    consecutive_failures = 0
    
    while True:
        try:
            settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
            sensing_enabled = settings.get("wifi_sensing_enabled", False)
            if not sensing_enabled:
                time.sleep(10)
                continue
                
            method = settings.get("wifi_sensing_method", "ping")
            target_ip = settings.get("wifi_sensing_target_ip", "").strip()
            target_mac = settings.get("wifi_sensing_target_mac", "").strip()
            
            is_detected = False
            
            if method == "ping":
                if target_ip:
                    res = subprocess.run(["arping", "-c", "1", "-w", "1", "-I", "wlo1", target_ip], capture_output=True)
                    if res.returncode == 0:
                        is_detected = True
                    else:
                        res2 = subprocess.run(["ping", "-c", "1", "-W", "1", target_ip], capture_output=True)
                        if res2.returncode == 0:
                            is_detected = True
            elif method == "sniff":
                if target_mac:
                    target_mac_lower = target_mac.lower()
                    detected_macs = []
                    def packet_handler(pkt):
                        try:
                            if pkt.haslayer(Dot11ProbeReq):
                                mac = pkt.addr2
                                if mac and mac.lower() == target_mac_lower:
                                    detected_macs.append(mac)
                        except Exception:
                            pass
                                
                    try:
                        from scapy.all import sniff, Dot11ProbeReq
                        sniff(iface="wlo1", prn=packet_handler, timeout=3, store=0)
                        if detected_macs:
                            is_detected = True
                    except Exception as ex:
                        print(f"Sniffing failed: {ex}")
                        
            current_time = time.time()
            if is_detected:
                consecutive_failures = 0
                if last_state != "present":
                    # Transition to Present!
                    away_duration = current_time - last_seen_time
                    away_min = away_duration / 60.0
                    
                    welcome_vol = 50
                    welcome_msg = ""
                    short_mode = False
                    
                    if last_state != "unknown":
                        if away_duration >= 3600: # 1 hour
                            welcome_vol = 80
                            welcome_msg = "Selamat datang kembali! Senang melihat Anda kembali setelah sekian lama."
                            short_mode = False
                        elif away_duration < 900: # 15 minutes
                            welcome_vol = 40
                            welcome_msg = "Zex standby."
                            short_mode = True
                        else: # Between 15m and 1h
                            welcome_vol = 55
                            welcome_msg = "Halo! Selamat datang kembali."
                            short_mode = False
                            
                        # Motion report addition
                        if motion_detected_while_away:
                            welcome_msg += f" Oh iya, ada gerakan terdeteksi di kamar pada pukul {motion_detected_timestamp}."
                            # Reset motion flags after reporting
                            motion_detected_while_away = False
                            motion_detected_timestamp = None
                            
                        # Set system volume
                        set_system_volume(welcome_vol)
                        
                        # Set response mode
                        if short_mode:
                            settings["response_mode"] = "short"
                            write_json_file(SETTINGS_FILE, settings)
                            
                        add_log("WIFI_SENSING_ARRIVED", "WiFiSensing", f"User datang. Durasi pergi: {away_min:.1f} m. Volume: {welcome_vol}%")
                        
                        # Make Mina speak
                        speak_sync(welcome_msg)
                        
                    last_state = "present"
                    user_presence_state = "present"
                last_seen_time = current_time
            else:
                consecutive_failures += 1
                if consecutive_failures >= 18:
                    if last_state != "absent":
                        add_log("WIFI_SENSING_DEPARTED", "WiFiSensing", "User terdeteksi pergi.")
                        last_state = "absent"
                        user_presence_state = "absent"

                        
        except Exception as e:
            print(f"Error in wifi_sensing_loop: {e}")
            
        time.sleep(10)

# wifi_sensing_thread = threading.Thread(target=wifi_sensing_loop, daemon=True)
# wifi_sensing_thread.start()


# --- CCTV STREAMING FUNCTIONS ---

def generate_placeholder(text="CCTV NONAKTIF"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 27, 24)
    cv2.circle(img, (320, 180), 45, (45, 41, 38), -1)
    cv2.circle(img, (320, 180), 30, (80, 75, 70), 3)
    cv2.circle(img, (320, 180), 12, (200, 80, 80), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = 300
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (150, 150, 150), thickness, cv2.LINE_AA)
    
    subtext = "app.gunturafandy.my.id"
    sub_size = cv2.getTextSize(subtext, font, 0.5, 1)[0]
    cv2.putText(img, subtext, ((640 - sub_size[0]) // 2, 340), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

def upload_to_immich_background(filename: str, is_video: bool = False):
    """Upload photo/video to Immich API in a background thread."""
    def run():
        try:
            import requests
            import hashlib
            import datetime
            settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
            sync_enabled = settings.get("immich_sync_enabled", False)
            api_key = settings.get("immich_api_key", "").strip()
            address = settings.get("immich_address", "http://127.0.0.1:2283").strip()
            if not sync_enabled or not api_key:
                return
            
            local_dir = VIDEOS_DIR if is_video else SNAPSHOTS_DIR
            local_path = os.path.join(local_dir, filename)
            if not os.path.exists(local_path):
                add_log("IMMICH_SYNC_ERROR", "CCTV", f"Berkas tidak ditemukan: {filename}")
                return
                
            upload_url = f"{address.rstrip('/')}/api/assets"
            headers = {
                "x-api-key": api_key,
                "Accept": "application/json"
            }
            
            dev_asset_id = hashlib.md5(filename.encode()).hexdigest()
            now_iso = datetime.datetime.utcnow().isoformat() + "Z"
            
            data = {
                "deviceAssetId": dev_asset_id,
                "deviceId": "cctv-mina",
                "fileCreatedAt": now_iso,
                "fileModifiedAt": now_iso
            }
            
            with open(local_path, "rb") as f:
                files = {"assetData": (filename, f)}
                r = requests.post(upload_url, headers=headers, data=data, files=files, timeout=45)
                
            if r.status_code in [200, 201]:
                add_log("IMMICH_SYNC_SUCCESS", "CCTV", f"Berhasil sinkronisasi {filename} ke Immich")
            else:
                add_log("IMMICH_SYNC_ERROR", "CCTV", f"Gagal sinkronisasi ({r.status_code}): {r.text[:200]}")
        except Exception as e:
            add_log("IMMICH_SYNC_ERROR", "CCTV", f"Error sinkronisasi: {str(e)}")
            
    threading.Thread(target=run, daemon=True).start()


def save_snapshot(frame):

    try:
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(SNAPSHOTS_DIR) if f.startswith("snap_") and f.endswith(".jpg")])
        if len(files) >= 50:
            for i in range(len(files) - 50 + 1):
                try:
                    os.remove(os.path.join(SNAPSHOTS_DIR, files[i]))
                except Exception:
                    pass
        ts_filename = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
        fpath = os.path.join(SNAPSHOTS_DIR, ts_filename)
        cv2.imwrite(fpath, frame)
        add_log("SNAPSHOT_SAVED", "CCTV", f"Mengambil foto otomatis: {ts_filename}")
        print(f"Periodic snapshot saved: {ts_filename}")
        upload_to_immich_background(ts_filename, is_video=False)
    except Exception as e:
        print(f"Error saving snapshot: {e}")

def get_current_cctv_fps() -> int:
    try:
        settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
        fps_day = int(settings.get("cctv_fps_day", 24))
        fps_night = int(settings.get("cctv_fps_night", 10))
        start_hour = int(settings.get("cctv_fps_night_start_hour", 0))
        end_hour = int(settings.get("cctv_fps_night_end_hour", 4))
        
        current_hour = datetime.datetime.now().hour
        
        is_night = False
        if start_hour <= end_hour:
            if start_hour <= current_hour < end_hour:
                is_night = True
        else:
            if current_hour >= start_hour or current_hour < end_hour:
                is_night = True
                
        return fps_night if is_night else fps_day
    except Exception:
        return 24

def camera_thread_func():
    global cap, camera_active, latest_frame, is_recording_video
    global user_presence_state, motion_detected_while_away, motion_detected_timestamp, last_camera_motion_time
    last_snapshot_time = 0
    
    static_back = None
    frame_count_since_reset = 0
    motion_trigger_cooldown = 0
    
    while True:
        try:
            settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
            motion_enabled = settings.get("cctv_motion_detection_enabled", False)
            sensing_enabled = settings.get("wifi_sensing_enabled", False)
            sensing_method = settings.get("wifi_sensing_method", "ping")
        except Exception:
            motion_enabled = False
            sensing_enabled = False
            sensing_method = "ping"
            
        run_motion = motion_enabled and (user_presence_state == "absent")
        run_camera_presence = sensing_enabled and (sensing_method == "camera")
        
        with camera_lock:
            active = camera_active
            recording = is_recording_video
        
        if active and not recording:
            if cap is None:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Failed to open camera /dev/video0")
                    cap = None
                    time.sleep(2)
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, frame = cap.read()
            if ret:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                _, jpeg = cv2.imencode('.jpg', frame)
                latest_frame = jpeg.tobytes()
                
                # Run background motion/presence check at 2 FPS if no clients are streaming
                is_background_mode = (run_motion or run_camera_presence) and (active_camera_clients == 0)
                if is_background_mode:
                    try:

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (21, 21), 0)
                        
                        baseline_path = os.path.join(DATA_DIR, "baseline_empty_calc.png")
                        if os.path.exists(baseline_path):
                            static_back = cv2.imread(baseline_path, cv2.IMREAD_GRAYSCALE)
                            frame_count_since_reset = 0
                        else:
                            if static_back is None or frame_count_since_reset >= 30:
                                static_back = gray
                                frame_count_since_reset = 0
                                
                        if static_back is not None:
                            diff_frame = cv2.absdiff(static_back, gray)
                            thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
                            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
                            contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            motion_detected = False
                            for contour in contours:
                                if cv2.contourArea(contour) >= 8000:
                                    motion_detected = True
                                    break
                                    
                            if motion_detected:
                                current_time = time.time()
                                
                                # 1. Camera-Only Presence Detection
                                if run_camera_presence:
                                    still_duration = current_time - last_camera_motion_time
                                    
                                    # Load dynamic thresholds from settings
                                    stillness_thresh = settings.get("presence_stillness_threshold_min", 3.0) * 60.0
                                    short_limit = settings.get("presence_short_limit_min", 15.0) * 60.0
                                    medium_limit = settings.get("presence_medium_limit_min", 60.0) * 60.0
                                    long_limit = settings.get("presence_long_limit_min", 120.0) * 60.0
                                    
                                    # Greet only if room was quiet/still for at least stillness_thresh seconds
                                    if still_duration >= stillness_thresh:
                                        welcome_vol = 50
                                        welcome_msg = ""
                                        short_mode = False
                                        
                                        if still_duration < short_limit:
                                            welcome_vol = 35
                                            welcome_msg = "Halo tuan."
                                            short_mode = True
                                        elif still_duration < medium_limit:
                                            welcome_vol = 50
                                            welcome_msg = "Halo, selamat datang kembali."
                                            short_mode = False
                                        elif still_duration < long_limit:
                                            welcome_vol = 65
                                            welcome_msg = "Selamat datang kembali. Anda sudah kembali di kamar."
                                            short_mode = False
                                        else:
                                            welcome_vol = 80
                                            welcome_msg = "Selamat datang kembali! Senang melihat Anda kembali setelah sekian lama."
                                            short_mode = False
                                            
                                        set_system_volume(welcome_vol)

                                        
                                        # Save settings response mode
                                        try:
                                            settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
                                            settings["response_mode"] = "short" if short_mode else "normal"
                                            write_json_file(SETTINGS_FILE, settings)
                                        except Exception:
                                            pass
                                            
                                        add_log("CAMERA_SENSING_ARRIVED", "CameraSensing", f"User terdeteksi kembali (CCTV). Diam: {still_duration/60.0:.1f} m. Volume: {welcome_vol}%")
                                        speak_sync(welcome_msg)
                                        
                                    last_camera_motion_time = current_time
                                
                                # 2. CCTV Security Alarm (when run_motion is active)
                                if run_motion:
                                    if current_time >= motion_trigger_cooldown:
                                        motion_trigger_cooldown = current_time + 60
                                        motion_detected_while_away = True
                                        motion_detected_timestamp = time.strftime("%H:%M")
                                        
                                        save_snapshot(frame)
                                        add_log("CCTV_MOTION_ALERT", "CCTV", f"Gerakan terdeteksi pada pukul {motion_detected_timestamp}")
                                        print(f"Motion detected at {motion_detected_timestamp}! Snapshot saved.")
                                        
                        frame_count_since_reset += 1
                        time.sleep(0.5)
                    except Exception as ex:
                        print(f"Error in motion detection algorithm: {ex}")
                else:
                    try:
                        settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
                        enabled = settings.get("cctv_snapshot_enabled", False)
                        interval_min = settings.get("cctv_snapshot_interval", 10)
                        
                        if enabled:
                            now = time.time()
                            if now - last_snapshot_time >= (interval_min * 60):
                                save_snapshot(frame)
                                last_snapshot_time = now
                    except Exception as e:
                        print(f"Error checking snapshot: {e}")
                    time.sleep(max(0.01, 1.0 / get_current_cctv_fps()))
            else:
                latest_frame = None
                time.sleep(0.1)
        else:
            if not active and cap is not None:
                cap.release()
                cap = None
                latest_frame = None
                static_back = None
            time.sleep(0.1)


# camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
# camera_thread.start()
def frame_generator():

    global latest_frame, camera_active
    while True:
        if camera_active:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                placeholder = generate_placeholder("LOADING CAMERA...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(max(0.01, 1.0 / get_current_cctv_fps()))
        else:
            placeholder = generate_placeholder("CCTV NONAKTIF")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.5)

# --- MUSIC PLAYER MODULE ---

def stop_music_play() -> str:
    global active_music_process, current_playing_name
    if active_music_process:
        try:
            active_music_process.terminate()
            active_music_process.wait(timeout=0.5)
        except Exception:
            try:
                active_music_process.kill()
            except Exception:
                pass
        active_music_process = None
        current_playing_name = None
        return "Musik dimatikan."
    return "Nggak ada musik."

def is_quiet_hours() -> bool:
    prefs = read_json_file(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    if not prefs.get("quiet_hours_enabled", True):
        return False
    current = datetime.datetime.now()
    
    def parse_clock(val):
        m = re.match(r"^(\d{1,2}):(\d{2})$", val.strip())
        return (int(m.group(1)), int(m.group(2))) if m else (0,0)
        
    start_h, start_m = parse_clock(prefs.get("quiet_hours_start", "22:00"))
    end_h, end_m = parse_clock(prefs.get("quiet_hours_end", "06:00"))
    
    now_min = current.hour * 60 + current.minute
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= now_min < end_min
    return now_min >= start_min or now_min < end_min

def record_music_play(play_type: str, detail: str):
    global PLAYED_MUSIC_HISTORY
    entry = {"type": play_type, "detail": detail}
    if not PLAYED_MUSIC_HISTORY or PLAYED_MUSIC_HISTORY[-1] != entry:
        PLAYED_MUSIC_HISTORY.append(entry)
        PLAYED_MUSIC_HISTORY = PLAYED_MUSIC_HISTORY[-20:]
        print(f"Recorded music history: {entry}")

def get_previous_music_entry():
    global PLAYED_MUSIC_HISTORY, active_music_process
    if not PLAYED_MUSIC_HISTORY:
        return None
    if active_music_process is not None and len(PLAYED_MUSIC_HISTORY) >= 2:
        return PLAYED_MUSIC_HISTORY[-2]
    return PLAYED_MUSIC_HISTORY[-1]

def play_music_stream(genre: str, *, user_requested: bool = True) -> str:
    global active_music_process, current_playing_name
    if not user_requested and is_quiet_hours():
        return "Jam tenang, musik dilewati."
    stop_music_play()
    genre_clean = (genre or "lofi").lower().strip()
    url = MUSIC_STREAMS.get(genre_clean, MUSIC_STREAMS["lofi"])
    try:
        mpv_cmd = ["mpv", "--no-video"]
        bt_sink = get_active_bt_sink()
        if bt_sink:
            mpv_cmd += [f"--audio-device=pulse/{bt_sink}"]
        mpv_cmd.append(url)
        active_music_process = subprocess.Popen(mpv_cmd)
        current_playing_name = f"Radio {genre_clean}"
        record_music_play("stream", genre_clean)
        return f"Radio {genre_clean} diputar."
    except Exception as e:
        return f"Gagal memutar radio: {e}"

def play_local_music(filename: str) -> str:
    global active_music_process, current_playing_name
    import difflib
    
    filepath = os.path.join(MUSIC_DIR, filename)
    if not os.path.exists(filepath):
        choices = sorted([f for f in os.listdir(MUSIC_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_MUSIC_EXTS])
        matched_name = None
        
        # 1. Substring matching (case-insensitive)
        for name in choices:
            if filename.lower() in name.lower():
                matched_name = name
                break
                
        # 2. Fuzzy matching (case-insensitive close matches)
        if not matched_name and choices:
            choices_no_ext = {os.path.splitext(c)[0].lower(): c for c in choices}
            close_matches = difflib.get_close_matches(filename.lower(), list(choices_no_ext.keys()), n=1, cutoff=0.4)
            if close_matches:
                matched_name = choices_no_ext[close_matches[0]]
                
        if matched_name:
            filepath = os.path.join(MUSIC_DIR, matched_name)
            filename = matched_name
        else:
            return "File musik tidak ditemukan."
            
    stop_music_play()
    try:
        mpv_cmd = ["mpv", "--no-video"]
        bt_sink = get_active_bt_sink()
        if bt_sink:
            mpv_cmd += [f"--audio-device=pulse/{bt_sink}"]
        mpv_cmd.append(filepath)
        active_music_process = subprocess.Popen(mpv_cmd)
        current_playing_name = filename
        record_music_play("local", filename)
        return f"Lagu '{filename}' diputar."
    except Exception as e:
        return f"Gagal memutar file: {e}"

def play_youtube_audio(query: str, *, user_requested: bool = True) -> str:
    global active_music_process, current_playing_name
    if not user_requested and is_quiet_hours():
        return "Jam tenang, musik dilewati."
    stop_music_play()
    try:
        mpv_cmd = ["mpv", "--no-video", "--ytdl-format=bestaudio"]
        bt_sink = get_active_bt_sink()
        if bt_sink:
            mpv_cmd += [f"--audio-device=pulse/{bt_sink}"]
        mpv_cmd.append(f"ytdl://ytsearch1:{query}")
        active_music_process = subprocess.Popen(
            mpv_cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        current_playing_name = f"YT: {query}"
        record_music_play("youtube", query)
        return f"Mencari dan memutar '{query}' dari YouTube..."
    except Exception as e:
        return f"Gagal memutar YouTube: {e}"

# --- VOICE ASSISTANT INTRINSICS & GEMINI ---

async def speak(text: str):
    ts = int(time.time() * 1000)
    wav_path = os.path.join(UPLOAD_DIR, f"tts_{ts}.wav")
    
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    tts_engine = settings.get("tts_engine", "edge-tts")
    voice_setting = settings.get("tts_voice", "female")
    
    if tts_engine == "piper":
        piper_bin = "/app/bin/piper/piper"
        model_path = "/app/models/id_ID-news_tts-medium.onnx"
        
        if not os.path.exists(piper_bin) or not os.path.exists(model_path):
            # Fallback path check
            piper_bin = os.path.join(os.path.dirname(__file__), "bin", "piper", "piper")
            model_path = os.path.join(os.path.dirname(__file__), "models", "id_ID-news_tts-medium.onnx")
            
        if os.path.exists(piper_bin) and os.path.exists(model_path):
            try:
                temp_wav = os.path.join(UPLOAD_DIR, f"tts_{ts}_temp.wav")
                
                # Execute Piper TTS process using asyncio with tuned speed & silences
                proc = await asyncio.create_subprocess_exec(
                    piper_bin, "--model", model_path, "--output_file", temp_wav,
                    "--length_scale", "1.12",
                    "--sentence_silence", "0.35",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate(input=text.encode('utf-8'))
                
                if os.path.exists(temp_wav):
                    if voice_setting == "male":
                        # Pitch shift to male voice using rubberband for high quality
                        proc_ffmpeg = await asyncio.create_subprocess_exec(
                            "ffmpeg", "-y", "-i", temp_wav, "-af", "rubberband=pitch=0.82", wav_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await proc_ffmpeg.communicate()
                        if os.path.exists(temp_wav):
                            try:
                                os.remove(temp_wav)
                            except Exception:
                                pass
                    else:
                        os.rename(temp_wav, wav_path)
                else:
                    print(f"Piper failed to output wave file: {stderr.decode('utf-8', errors='ignore')}")
                    tts_engine = "edge-tts"
            except Exception as e:
                print(f"Piper exception: {e}. Falling back to Edge-TTS.")
                tts_engine = "edge-tts"
        else:
            print(f"Piper binary or model missing. Falling back to Edge-TTS.")
            tts_engine = "edge-tts"
            
    if tts_engine != "piper":
        mp3_path = os.path.join(UPLOAD_DIR, f"tts_{ts}.mp3")
        voice_id = "id-ID-GadisNeural" if voice_setting == "female" else "id-ID-ArdiNeural"
        
        try:
            import edge_tts
            comm = edge_tts.Communicate(text, voice_id)
            await comm.save(mp3_path)
        except Exception as e:
            print(f"Edge TTS failed: {e}. Falling back to gTTS.")
            try:
                from gtts import gTTS
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: gTTS(text=text, lang="id").save(mp3_path))
            except Exception as ex:
                print(f"gTTS failed: {ex}")
                return
                
        try:
            subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "16000", wav_path], capture_output=True, text=True)
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
            return
            
    async with audio_lock:
        stop_active_audio()
        # Set system volume dynamically to match volume slider/settings
        vol = settings.get("volume")
        if vol is None:
            vol = get_configured_default_volume()
        set_system_volume(vol)
        
        bt_sink = get_active_bt_sink()
        if os.environ.get("PULSE_SERVER") or os.path.exists("/tmp/pulse-socket"):
            if bt_sink:
                cmd = ["paplay", "--device", bt_sink, wav_path]
            else:
                cmd = ["paplay", wav_path]
        else:
            cmd = ["aplay", "-q", wav_path]
        try:
            global active_play_process
            active_play_process = subprocess.Popen(cmd)
        except Exception as ex:
            print(f"Failed to play TTS wav: {ex}")

# Local Intent Fallback Routing
def parse_time_expression(text: str) -> tuple[Optional[str], Optional[str]]:
    text = text.lower().strip()
    m = re.search(r"jam\s*(\d{1,2})(?:[:.](\d{1,2}))?\s*(pagi|siang|sore|malam)?", text)
    if not m:
        return None, None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = m.group(3)
    if suffix == "siang" and hour < 12:
        hour += 12
    elif suffix == "sore" and hour < 12:
        hour += 12
    elif suffix == "malam" and hour < 12:
        hour = 0 if hour == 12 else hour + 12
    elif suffix == "pagi" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None, None
    return f"{hour:02d}:{minute:02d}", m.group(0)

def parse_relative_minutes(text: str) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*menit\s+lagi", text.lower())
    return int(m.group(1)) if m else None

def parse_everyday_repeat(text: str) -> list:
    lowered = text.lower()
    if "setiap hari" in lowered:
        return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    res = []
    for d_id, d_en in [("senin", "Mon"), ("selasa", "Tue"), ("rabu", "Wed"), ("kamis", "Thu"), ("jumat", "Fri"), ("sabtu", "Sat"), ("minggu", "Sun")]:
        if f"setiap {d_id}" in lowered:
            res.append(d_en)
    return res

def extract_reminder_message(text: str) -> str:
    lowered = text.lower()
    message = text
    for prefix in ["ingatkan", "ingetin", "tolong ingetin", "tolong ingatkan", "set reminder", "buat reminder"]:
        idx = lowered.find(prefix)
        if idx != -1:
            message = text[idx + len(prefix):].strip(" ,")
            break
    message = re.sub(r"^aku\s+", "", message, flags=re.I)
    message = re.sub(r"^buat\s+", "", message, flags=re.I)
    message = re.sub(r"^tentang\s+", "", message, flags=re.I)
    message = re.sub(r"\bsetiap\s+(hari|senin|selasa|rabu|kamis|jumat|sabtu|minggu)\b", "", message, flags=re.I)
    message = re.sub(r"\bjam\s*\d{1,2}(?:[:.]\d{1,2})?\s*(pagi|siang|sore|malam)?\b", "", message, flags=re.I)
    message = re.sub(r"\b\d+\s*menit\s+lagi\b", "", message, flags=re.I)
    return message.strip(" ,-.") or "ingat sesuatu"

async def create_local_alarm(name: str, alarm_time: str, tts_text: Optional[str] = None, repeat_days: Optional[List[str]] = None) -> dict:
    alarms = get_alarms()
    new_alarm = {
        "id": str(uuid.uuid4()),
        "name": name or "Alarm",
        "time": alarm_time,
        "type": "tts",
        "tts_text": tts_text or f"Sekarang jam {alarm_time}.",
        "tts_lang": "id-ID-GadisNeural",
        "ringtone": "",
        "repeat_days": repeat_days or [],
        "volume": 80,
        "enabled": True,
        "snooze_duration": 5,
        "last_triggered": ""
    }
    alarms.append(new_alarm)
    save_alarms(alarms)
    add_log("CREATE", new_alarm["name"], f"Alarm diset untuk {alarm_time}")
    return new_alarm

async def create_local_timer(minutes: int, label: str = "Timer") -> dict:
    target = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    alarm_time = target.strftime("%H:%M")
    name = label or "Timer"
    return await create_local_alarm(name, alarm_time, f"Timer {name} selesai!")

async def get_briefing_text() -> str:
    prefs = read_json_file(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    city = prefs.get("default_city", "Malang")
    weather = "Gagal mengambil data cuaca."
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(f"https://wttr.in/{city}?format=3")
            if r.status_code == 200:
                weather = f"Cuaca di {r.text.strip()}"
    except Exception:
        pass
        
    alarms = [a for a in get_alarms() if a.get("enabled")]
    if alarms:
        alarm_text = ", ".join(f"{a.get('name')} jam {a.get('time')}" for a in alarms[:3])
    else:
        alarm_text = "tidak ada alarm aktif"
        
    todos = [t for t in read_json_file(TODO_FILE, []) if not t.get("done")]
    todo_text = f"{len(todos)} tugas tertunda" if todos else "tidak ada tugas tertunda"
    
    return f"Briefing hari ini: {weather}. Alarm aktif: {alarm_text}. To-do list: {todo_text}."

def trigger_manual_snapshot() -> Optional[str]:
    global cap, latest_frame
    try:
        opened_here = False
        temp_cap = cap
        if temp_cap is None:
            temp_cap = cv2.VideoCapture(0)
            opened_here = True
            time.sleep(0.5) # let camera adjust
            
        if temp_cap is not None and temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                ts_filename = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
                fpath = os.path.join(SNAPSHOTS_DIR, ts_filename)
                cv2.imwrite(fpath, frame)
                if opened_here:
                    temp_cap.release()
                return ts_filename
        if opened_here and temp_cap is not None:
            temp_cap.release()
    except Exception as e:
        print(f"Error in trigger_manual_snapshot: {e}")
    return None

def play_discord_join_sound():
    try:
        sound_path = os.path.join(UPLOAD_DIR, "discord-join.mp3")
        if os.path.exists(sound_path):
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-volume", "35", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error playing discord join sound: {e}")

def camera_on() -> str:
    global camera_active
    with camera_lock:
        if not camera_active:
            play_discord_join_sound()
        camera_active = True
    return "Kamera dinyalakan."

def camera_off() -> str:
    global camera_active
    with camera_lock:
        camera_active = False
    return "Kamera dimatikan."

def smart_snapshot_with_warmup(warmup_sec: float = 2.0) -> Optional[str]:
    """Turn camera on if needed, wait for warm-up, take snapshot, return filename."""
    global cap, camera_active, latest_frame
    
    was_off = not camera_active
    if was_off:
        camera_on()  # Permanently turn on camera
        time.sleep(warmup_sec)  # wait 2 seconds for camera thread to initialize & auto-expose
        
    # If camera is already active, we just save the latest_frame bytes directly
    if camera_active and latest_frame is not None:
        try:
            ts_filename = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
            fpath = os.path.join(SNAPSHOTS_DIR, ts_filename)
            with open(fpath, "wb") as f:
                f.write(latest_frame)
            add_log("SNAPSHOT_SAVED", "CCTV", f"Mengambil foto manual (instant): {ts_filename}")
            upload_to_immich_background(ts_filename, is_video=False)
            return ts_filename
        except Exception as e:
            print(f"Error saving instant snapshot: {e}")
            
    # Fallback to manual capture if latest_frame is None
    try:
        temp_cap = cv2.VideoCapture(0)
        if temp_cap is not None and temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Actively read and discard 30 frames to let auto-exposure adjust
            frame = None
            for _ in range(30):
                ret, f = temp_cap.read()
                if ret:
                    frame = f
                time.sleep(0.05)
                
            if frame is not None:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                ts_filename = time.strftime("snap_%Y%m%d_%H%M%S.jpg")
                fpath = os.path.join(SNAPSHOTS_DIR, ts_filename)
                cv2.imwrite(fpath, frame)
                temp_cap.release()
                add_log("SNAPSHOT_SAVED", "CCTV", f"Mengambil foto manual (warmup): {ts_filename}")
                upload_to_immich_background(ts_filename, is_video=False)
                return ts_filename
            temp_cap.release()
    except Exception as e:
        print(f"Error in smart_snapshot_with_warmup: {e}")
    return None



def record_video(duration_sec: int = 5) -> Optional[str]:
    """Record video for duration_sec seconds, return filename or None."""
    global cap, camera_active, is_recording_video
    if is_recording_video:
        return None
    
    was_off = not camera_active
    if was_off:
        camera_on()  # Permanently turn on camera
        time.sleep(2.0)  # wait 2 seconds for camera thread to initialize & auto-expose
        
    try:
        is_recording_video = True
        # Let the camera thread yield control
        time.sleep(0.15)
        
        opened_here = False
        temp_cap = cap
        if temp_cap is None or not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(0)
            opened_here = True
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Warm up by reading and discarding 30 frames
            for _ in range(30):
                temp_cap.read()
                time.sleep(0.05)
        
        if temp_cap is None or not temp_cap.isOpened():
            return None
        
        ts_filename = time.strftime("video_%Y%m%d_%H%M%S.mp4")
        temp_fpath = os.path.join(VIDEOS_DIR, "raw_" + ts_filename)
        final_fpath = os.path.join(VIDEOS_DIR, ts_filename)
        
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 30.0
            
        total_frames = int(fps * duration_sec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_fpath, fourcc, fps, (640, 480))
        
        frames_written = 0
        while frames_written < total_frames:
            ret, frame = temp_cap.read()
            if ret:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                out.write(frame)
                frames_written += 1
            else:
                time.sleep(0.01)
        
        out.release()
        if opened_here or temp_cap is not cap:
            temp_cap.release()
            
        # Convert raw video to H.264 mp4 using FFmpeg for browser playback compatibility
        try:
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", temp_fpath, "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_fpath]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                if os.path.exists(temp_fpath):
                    os.remove(temp_fpath)
            else:
                print(f"FFmpeg video conversion failed: {res.stderr}")
                os.rename(temp_fpath, final_fpath)
        except Exception as e:
            print(f"Error converting video with FFmpeg: {e}")
            if os.path.exists(temp_fpath):
                os.rename(temp_fpath, final_fpath)
        
        # Auto prune old videos (keep max 20)
        try:
            vfiles = sorted([f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")])
            while len(vfiles) > 20:
                os.remove(os.path.join(VIDEOS_DIR, vfiles.pop(0)))
        except Exception:
            pass
        except Exception:
            pass
        
        upload_to_immich_background(ts_filename, is_video=True)
        return ts_filename
    except Exception as e:
        print(f"Error recording video: {e}")
        return None
    finally:
        is_recording_video = False


VOLUME_WORDS = {
    "lembut": 20, "pelan": 20, "kecil": 20,
    "sedang": 45, "medium": 45,
    "normal": 60,
    "agak keras": 70, "sedikit keras": 70,
    "keras": 80, "kencang": 80,
    "full": 100, "maksimal": 100,
}

def parse_volume_modifier(text: str) -> Optional[int]:
    lower = text.lower()
    m = re.search(r'(?:volume\s*)?(?:jadi\s*|ke\s*)?(\d{1,3})\s*%', lower)
    if m:
        return max(0, min(100, int(m.group(1))))
    m = re.search(r'(?:volume|vol)\s*(?:jadi|ke)?\s*(\d{1,3})(?!\s*%)', lower)
    if m:
        return max(0, min(100, int(m.group(1))))
    for word, level in sorted(VOLUME_WORDS.items(), key=lambda x: -len(x[0])):
        if word in lower:
            return level
    return None

def parse_delayed_instruction(text: str) -> Optional[tuple]:
    """Parse text for delay instructions like 'dalam 10 detik' or 'dalam 5 menit'.
    Returns (delay_seconds, clean_text, time_unit_str) or None.
    """
    lowered = text.lower()
    # Check seconds
    m = re.search(r'dalam\s+(\d+)\s*(?:detik|det|second|s\b)', lowered)
    if m:
        sec = float(m.group(1))
        clean = re.sub(r'dalam\s+\d+\s*(?:detik|det|second|s\b)', '', text, flags=re.IGNORECASE).strip()
        return sec, clean, f"{int(sec)} detik"
    # Check minutes
    m = re.search(r'dalam\s+(\d+)\s*(?:menit|men|minute|m\b)', lowered)
    if m:
        val = int(m.group(1))
        sec = float(val) * 60.0
        clean = re.sub(r'dalam\s+\d+\s*(?:menit|men|minute|m\b)', '', text, flags=re.IGNORECASE).strip()
        return sec, clean, f"{val} menit"
    return None

async def delayed_command_executor(delay_sec: float, action: str, args: dict):
    await asyncio.sleep(delay_sec)
    try:
        if action == "stop_music":
            stop_music_play()
            add_log("DELAYED_ACTION", "System", "Musik dimatikan otomatis via timer.")
        elif action == "set_volume":
            level = int(args.get("level", 50))
            set_system_volume(level)
            add_log("DELAYED_ACTION", "System", f"Volume disetel ke {level}% otomatis via timer.")
        elif action == "camera_off":
            camera_off()
            add_log("DELAYED_ACTION", "System", "Kamera dinonaktifkan otomatis via timer.")
        elif action == "camera_on":
            camera_on()
            add_log("DELAYED_ACTION", "System", "Kamera diaktifkan otomatis via timer.")
        elif action == "cctv_snapshot":
            smart_snapshot_with_warmup(warmup_sec=2.0)
            add_log("DELAYED_ACTION", "System", "Snapshot CCTV diambil otomatis via timer.")
        elif action == "play_music":
            play_music_stream(args.get("genre", "lofi"))
            add_log("DELAYED_ACTION", "System", f"Radio streaming {args.get('genre')} diputar otomatis via timer.")
        elif action == "play_local_music":
            play_local_music(args.get("filename", ""))
            add_log("DELAYED_ACTION", "System", f"Lagu lokal {args.get('filename')} diputar otomatis via timer.")
        elif action == "play_youtube":
            play_youtube_audio(args.get("query", ""))
            add_log("DELAYED_ACTION", "System", f"Lagu YouTube {args.get('query')} diputar otomatis via timer.")
    except Exception as e:
        print(f"Error in delayed command executor: {e}")

async def local_intent_router(text: str) -> Optional[dict]:
    lowered = text.lower().strip()
    if not lowered:
        return None
        
    # Check if there is a delayed instruction
    delay_info = parse_delayed_instruction(text)
    if delay_info:
        delay_sec, clean_text, time_str = delay_info
        # Call the core router logic with the cleaned command
        res = await _execute_local_intent(clean_text)
        if res and "action" in res:
            act = res["action"].get("action")
            # Only schedule supported actions
            supported = ["stop_music", "set_volume", "camera_on", "camera_off", "cctv_snapshot", "play_music", "play_local_music", "play_youtube"]
            if act in supported:
                # Schedule delayed task using asyncio
                asyncio.create_task(delayed_command_executor(delay_sec, act, res["action"]))
                
                # Format friendly confirmation message
                action_label = {
                    "stop_music": "mematikan musik",
                    "set_volume": f"menyetel volume ke {res['action'].get('level', 50)}%",
                    "camera_on": "menyalakan kamera CCTV",
                    "camera_off": "mematikan kamera CCTV",
                    "cctv_snapshot": "mengambil foto CCTV",
                    "play_music": f"memutar radio {res['action'].get('genre', 'lofi')}",
                    "play_local_music": f"memutar lagu {res['action'].get('filename', 'lokal')}",
                    "play_youtube": f"memutar YouTube {res['action'].get('query', 'lagu')}"
                }.get(act, "menjalankan aksi")
                
                return {
                    "reply": f"Baik, saya jadwalkan untuk {action_label} dalam {time_str}.",
                    "action": {"action": "delayed_action_scheduled", "delay": delay_sec, "target_action": act}
                }
        return None
        
    return await _execute_local_intent(text)

async def _execute_local_intent(text: str) -> Optional[dict]:
    lowered = text.lower().strip()
    if not lowered:
        return None
        
    # Check briefings
    if "briefing" in lowered:
        return {"reply": await get_briefing_text(), "action": {"action": "get_briefing"}}
        
    # Stop music
    if any(k in lowered for k in ["matiin musik", "hentikan musik", "stop musik", "berhentiin musik"]):
        return {"reply": stop_music_play(), "action": {"action": "stop_music"}}
        
    # Volume control
    if "volume" in lowered:
        m = re.search(r"volume\s*(?:jadi|ke)?\s*(\d{1,3})", lowered)
        if m:
            level = max(0, min(100, int(m.group(1))))
            set_system_volume(level)
            return {"reply": f"Volume disetel ke {level}%.", "action": {"action": "set_volume", "level": level}}
        if any(k in lowered for k in ["naikin", "besarin"]):
            level = min(100, get_system_volume() + 10)
            set_system_volume(level)
            return {"reply": f"Volume dinaikkan ke {level}%.", "action": {"action": "set_volume", "level": level}}
        if any(k in lowered for k in ["kecilin", "turunin"]):
            level = max(0, get_system_volume() - 10)
            set_system_volume(level)
            return {"reply": f"Volume diturunkan ke {level}%.", "action": {"action": "set_volume", "level": level}}
            
    # Play youtube/local/streams (with optional volume modifier)
    if lowered.startswith("putar "):
        query = text[6:].strip()
        vol_override = parse_volume_modifier(lowered)

        # Strip volume/modifier words from query before searching music title
        clean_query = re.sub(r'\s*(?:dengan|pake|pakai)?\s*(?:volume|vol)?\s*(?:jadi|ke)?\s*\d{1,3}\s*%?', '', query, flags=re.IGNORECASE).strip()
        for word in sorted(VOLUME_WORDS.keys(), key=len, reverse=True):
            clean_query = re.sub(rf'\b{re.escape(word)}\b', '', clean_query, flags=re.IGNORECASE).strip()
        clean_query = re.sub(r'\s{2,}', ' ', clean_query).strip() or query

        if vol_override is not None:
            set_system_volume(vol_override)
        vol_suffix = f" Volume disetel ke {vol_override}%." if vol_override is not None else ""

        if any(g in lowered for g in MUSIC_STREAMS.keys()):
            for g in MUSIC_STREAMS.keys():
                if g in lowered:
                    reply = play_music_stream(g)
                    return {"reply": reply + vol_suffix, "action": {"action": "play_music", "genre": g}}
        # Local music search
        res = play_local_music(clean_query)
        if "tidak ditemukan" not in res:
            return {"reply": res + vol_suffix, "action": {"action": "play_local_music", "filename": clean_query}}
        # Fallback YouTube
        yt_reply = play_youtube_audio(clean_query)
        return {"reply": yt_reply + vol_suffix, "action": {"action": "play_youtube", "query": clean_query}}
        
    # Set Alarm
    if any(k in lowered for k in ["alarm", "bangunkan"]) and not any(k in lowered for k in ["ingatkan", "ingetin"]):
        alarm_time, raw_expr = parse_time_expression(lowered)
        if alarm_time:
            name = "Alarm"
            m = re.search(r"alarm\s+(.+?)\s+jam", lowered)
            if m:
                name = m.group(1).strip().title()
            await create_local_alarm(name, alarm_time)
            return {"reply": f"Alarm '{name}' diset untuk jam {alarm_time}.", "action": {"action": "set_alarm", "time": alarm_time}}
            
    # Camera ON/OFF control
    if any(k in lowered for k in ["nyalakan kamera", "hidupkan kamera", "buka kamera", "aktifkan kamera", "kamera on", "on kamera"]):
        return {"reply": camera_on() + " Streaming CCTV aktif.", "action": {"action": "camera_on"}}
    if any(k in lowered for k in ["matikan kamera", "tutup kamera", "nonaktifkan kamera", "kamera off", "off kamera"]):
        return {"reply": camera_off() + " Streaming CCTV dimatikan.", "action": {"action": "camera_off"}}

    # Take CCTV Snapshot (with auto warm-up)
    if any(k in lowered for k in ["ambil foto", "tangkapan cctv", "tunjukkan cctv", "ambil cctv", "foto cctv", "tangkapan foto cctv", "potret cctv", "foto sekarang", "fotokan", "foto", "potret", "jepret", "ambil gambar"]):
        filename = smart_snapshot_with_warmup(warmup_sec=2.0)
        if filename:
            return {"reply": "Baik, kamera sudah siap. Ini foto tangkapan terbaru dari CCTV kamar Anda.", "action": {"action": "cctv_snapshot", "filename": filename, "url": f"/static/snapshots/{filename}"}}
        else:
            return {"reply": "Maaf, gagal mengakses kamera CCTV saat ini.", "action": {"action": "error"}}

    # Record Video
    if any(k in lowered for k in ["rekam video", "rekam sekarang", "videoin", "ambil video", "record video", "rekam kamera"]):
        m = re.search(r'(\d+)\s*(?:detik|det|second|s\b)', lowered)
        duration = int(m.group(1)) if m else 5
        duration = max(1, min(60, duration))  # clamp 1-60 detik
        filename = record_video(duration_sec=duration)
        if filename:
            return {"reply": f"Video {duration} detik selesai direkam.", "action": {"action": "video_recorded", "filename": filename, "url": f"/static/videos/{filename}", "duration": duration}}
        else:
            return {"reply": "Maaf, gagal merekam video. Mungkin kamera sedang sibuk.", "action": {"action": "error"}}

    # Play Previous Music
    if any(k in lowered for k in ["putar musik sebelumnya", "putar lagu sebelumnya", "putar lagu tadi", "putar musik tadi", "lagu sebelumnya", "musik sebelumnya"]):
        entry = get_previous_music_entry()
        if entry:
            t = entry["type"]
            d = entry["detail"]
            if t == "stream":
                reply = play_music_stream(d)
                return {"reply": f"Baik, memutar kembali radio streaming sebelumnya: {d}.", "action": {"action": "play_music", "genre": d}}
            elif t == "local":
                reply = play_local_music(d)
                return {"reply": f"Baik, memutar kembali lagu sebelumnya: {d}.", "action": {"action": "play_local_music", "filename": d}}
            elif t == "youtube":
                reply = play_youtube_audio(d)
                return {"reply": f"Baik, memutar kembali lagu YouTube sebelumnya: {d}.", "action": {"action": "play_youtube", "query": d}}
        else:
            return {"reply": "Belum ada riwayat musik yang terdeteksi sebelumnya.", "action": {"action": "error"}}

    return None

# Gemini Tool Declarations
ALARM_TOOL = {"name": "set_alarm", "description": "Setel alarm tidur untuk jam tertentu dengan nama alarm dan teks TTS kustom.", "parameters": {"type": "OBJECT", "properties": {"name": {"type": "STRING", "description": "Nama alarm."}, "time": {"type": "STRING", "description": "Waktu HH:MM format 24 jam."}, "tts_text": {"type": "STRING", "description": "Teks kustom untuk alarm."}}, "required": ["name", "time"]}}
WEATHER_TOOL = {"name": "get_weather", "description": "Ambil prakiraan cuaca kota tertentu.", "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}, "required": ["city"]}}
SEARCH_TOOL = {"name": "search_web", "description": "Cari informasi terbaru di internet.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}}
PLAY_MUSIC_TOOL = {"name": "play_music", "description": "Putar radio streaming berdasarkan genre. Bisa sekaligus atur volume (angka 0-100, atau kata: lembut/sedang/normal/keras).", "parameters": {"type": "OBJECT", "properties": {"genre": {"type": "STRING"}, "volume": {"type": "INTEGER", "description": "Volume 0-100. Opsional."}}, "required": ["genre"]}}
STOP_MUSIC_TOOL = {"name": "stop_music", "description": "Hentikan musik.", "parameters": {"type": "OBJECT", "properties": {}}}
LOCAL_PLAY_TOOL = {"name": "play_local_music", "description": "Putar file musik lokal dari perpustakaan. Bisa sekaligus atur volume.", "parameters": {"type": "OBJECT", "properties": {"filename": {"type": "STRING"}, "volume": {"type": "INTEGER", "description": "Volume 0-100. Opsional."}}, "required": ["filename"]}}
VOLUME_TOOL = {"name": "set_volume", "description": "Atur volume speaker 0 sampai 100.", "parameters": {"type": "OBJECT", "properties": {"level": {"type": "INTEGER"}}, "required": ["level"]}}
TIMER_TOOL = {"name": "set_timer", "description": "Buat timer countdown dalam menit.", "parameters": {"type": "OBJECT", "properties": {"minutes": {"type": "INTEGER"}, "label": {"type": "STRING"}}, "required": ["minutes"]}}
BRIEFING_TOOL = {"name": "get_briefing", "description": "Ambil briefing harian.", "parameters": {"type": "OBJECT", "properties": {}}}
YOUTUBE_TOOL = {"name": "play_youtube", "description": "Putar audio dari YouTube berdasarkan query. Bisa sekaligus atur volume.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}, "volume": {"type": "INTEGER", "description": "Volume 0-100. Opsional."}}, "required": ["query"]}}
SNAPSHOT_TOOL = {"name": "take_cctv_snapshot", "description": "Ambil tangkapan foto CCTV kamar saat ini dengan warm-up otomatis 2 detik.", "parameters": {"type": "OBJECT", "properties": {}}}
PREV_MUSIC_TOOL = {"name": "play_previous_music", "description": "Putar kembali musik atau lagu yang diputar sebelumnya.", "parameters": {"type": "OBJECT", "properties": {}}}
CAMERA_ON_TOOL = {"name": "camera_on", "description": "Nyalakan/aktifkan kamera CCTV kamar.", "parameters": {"type": "OBJECT", "properties": {}}}
CAMERA_OFF_TOOL = {"name": "camera_off", "description": "Matikan/nonaktifkan kamera CCTV kamar.", "parameters": {"type": "OBJECT", "properties": {}}}
RECORD_VIDEO_TOOL = {"name": "record_video", "description": "Rekam video dari kamera CCTV kamar selama beberapa detik. Default 5 detik, max 60 detik.", "parameters": {"type": "OBJECT", "properties": {"duration": {"type": "INTEGER", "description": "Durasi rekaman dalam detik (1-60). Default 5."}}, "required": []}}
SCHEDULE_TOOL = {"name": "schedule_delayed_command", "description": "Jadwalkan eksekusi perintah otomatis setelah beberapa detik atau menit. Contoh: matikan musik atau set volume.", "parameters": {"type": "OBJECT", "properties": {"delay_seconds": {"type": "INTEGER", "description": "Delay waktu tunggu dalam detik."}, "command_type": {"type": "STRING", "description": "Tipe perintah yang ingin dijadwalkan: stop_music, set_volume, camera_on, camera_off, cctv_snapshot"}, "command_args": {"type": "OBJECT", "description": "Argumen perintah, e.g. {'level': 20} untuk set_volume."}}, "required": ["delay_seconds", "command_type"]}}

ALL_TOOLS = [{"functionDeclarations": [ALARM_TOOL, WEATHER_TOOL, SEARCH_TOOL, PLAY_MUSIC_TOOL, STOP_MUSIC_TOOL, LOCAL_PLAY_TOOL, VOLUME_TOOL, TIMER_TOOL, BRIEFING_TOOL, YOUTUBE_TOOL, SNAPSHOT_TOOL, PREV_MUSIC_TOOL, CAMERA_ON_TOOL, CAMERA_OFF_TOOL, RECORD_VIDEO_TOOL, SCHEDULE_TOOL]}]

async def execute_function_call(func_name: str, args: dict) -> dict:
    if func_name == "set_alarm":
        time_str = args.get("time")
        name = args.get("name", "Alarm")
        tts = args.get("tts_text")
        if time_str:
            await create_local_alarm(name, time_str, tts)
            return {"reply": f"Alarm '{name}' diset untuk jam {time_str}.", "action": {"action": "set_alarm", "time": time_str}}
    elif func_name == "get_weather":
        city = args.get("city", "Malang")
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.get(f"https://wttr.in/{city}?format=3")
                if r.status_code == 200:
                    return {"reply": f"Cuaca saat ini: {r.text.strip()}.", "action": {"action": "get_weather"}}
        except Exception:
            pass
        return {"reply": f"Gagal mengambil cuaca untuk {city}.", "action": {"action": "get_weather"}}
    elif func_name == "search_web":
        query = args.get("query", "")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            async with httpx.AsyncClient(headers=headers, timeout=10.0, follow_redirects=True) as client:
                r = await client.get(f"https://html.duckduckgo.com/html/?q={query}")
                if r.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "html.parser")
                    snippets = [a.get_text().strip() for a in soup.find_all("a", class_="result__snippet")[:3]]
                    if snippets:
                        snippet_str = "\n".join(f"- {s}" for s in snippets)
                        return {"reply": f"Hasil internet:\n{snippet_str}", "action": {"action": "search_web"}}
        except Exception:
            pass
        return {"reply": f"Gagal mencari info tentang '{query}' di internet.", "action": {"action": "search_web"}}
    elif func_name == "play_music":
        genre = args.get("genre", "lofi")
        vol = args.get("volume")
        if vol is not None:
            set_system_volume(max(0, min(100, int(vol))))
        vol_suffix = f" Volume disetel ke {vol}%." if vol is not None else ""
        return {"reply": play_music_stream(genre) + vol_suffix, "action": {"action": "play_music", "genre": genre}}
    elif func_name == "play_local_music":
        filename = args.get("filename", "")
        vol = args.get("volume")
        if vol is not None:
            set_system_volume(max(0, min(100, int(vol))))
        vol_suffix = f" Volume disetel ke {vol}%." if vol is not None else ""
        return {"reply": play_local_music(filename) + vol_suffix, "action": {"action": "play_local_music", "filename": filename}}
    elif func_name == "stop_music":
        return {"reply": stop_music_play(), "action": {"action": "stop_music"}}
    elif func_name == "set_volume":
        level = int(args.get("level", 50))
        set_system_volume(level)
        return {"reply": f"Volume disetel ke {level}%.", "action": {"action": "set_volume", "level": level}}
    elif func_name == "set_timer":
        minutes = int(args.get("minutes", 5))
        label = args.get("label", "Timer")
        await create_local_timer(minutes, label)
        return {"reply": f"Timer '{label}' diset {minutes} menit lagi.", "action": {"action": "set_timer", "minutes": minutes}}
    elif func_name == "get_briefing":
        return {"reply": await get_briefing_text(), "action": {"action": "get_briefing"}}
    elif func_name == "play_youtube":
        query = args.get("query", "")
        vol = args.get("volume")
        if vol is not None:
            set_system_volume(max(0, min(100, int(vol))))
        vol_suffix = f" Volume disetel ke {vol}%." if vol is not None else ""
        return {"reply": play_youtube_audio(query) + vol_suffix, "action": {"action": "play_youtube", "query": query}}
    elif func_name == "take_cctv_snapshot":
        filename = smart_snapshot_with_warmup(warmup_sec=2.0)
        if filename:
            return {"reply": "Baik, kamera sudah siap. Ini foto tangkapan terbaru dari CCTV kamar Anda.", "action": {"action": "cctv_snapshot", "filename": filename, "url": f"/static/snapshots/{filename}"}}
        else:
            return {"reply": "Maaf, gagal mengakses kamera CCTV saat ini.", "action": {"action": "error"}}
    elif func_name == "play_previous_music":
        entry = get_previous_music_entry()
        if entry:
            t = entry["type"]
            d = entry["detail"]
            if t == "stream":
                reply = play_music_stream(d)
                return {"reply": f"Baik, memutar kembali radio streaming sebelumnya: {d}.", "action": {"action": "play_music", "genre": d}}
            elif t == "local":
                reply = play_local_music(d)
                return {"reply": f"Baik, memutar kembali lagu sebelumnya: {d}.", "action": {"action": "play_local_music", "filename": d}}
            elif t == "youtube":
                reply = play_youtube_audio(d)
                return {"reply": f"Baik, memutar kembali lagu YouTube sebelumnya: {d}.", "action": {"action": "play_youtube", "query": d}}
        else:
            return {"reply": "Belum ada riwayat musik yang terdeteksi sebelumnya.", "action": {"action": "error"}}
    elif func_name == "camera_on":
        return {"reply": camera_on() + " Streaming CCTV aktif.", "action": {"action": "camera_on"}}
    elif func_name == "camera_off":
        return {"reply": camera_off() + " Streaming CCTV dimatikan.", "action": {"action": "camera_off"}}
    elif func_name == "record_video":
        duration = max(1, min(60, int(args.get("duration", 5))))
        filename = record_video(duration_sec=duration)
        if filename:
            return {"reply": f"Video {duration} detik selesai direkam.", "action": {"action": "video_recorded", "filename": filename, "url": f"/static/videos/{filename}", "duration": duration}}
        else:
            return {"reply": "Maaf, gagal merekam video.", "action": {"action": "error"}}
            
    elif func_name == "schedule_delayed_command":
        delay = int(args.get("delay_seconds", 5))
        cmd = args.get("command_type", "")
        cmd_args = args.get("command_args", {})
        
        supported = ["stop_music", "set_volume", "camera_on", "camera_off", "cctv_snapshot"]
        if cmd in supported:
            asyncio.create_task(delayed_command_executor(float(delay), cmd, cmd_args))
            
            action_label = {
                "stop_music": "mematikan musik",
                "set_volume": f"menyetel volume ke {cmd_args.get('level', 50)}%",
                "camera_on": "menyalakan kamera CCTV",
                "camera_off": "mematikan kamera CCTV",
                "cctv_snapshot": "mengambil foto CCTV"
            }.get(cmd, "menjalankan aksi")
            
            time_unit = f"{delay} detik" if delay < 60 else f"{delay // 60} menit"
            return {
                "reply": f"Baik, saya jadwalkan untuk {action_label} dalam {time_unit}.",
                "action": {"action": "delayed_action_scheduled", "delay": delay, "target_action": cmd}
            }
        else:
            return {"reply": f"Maaf, tipe perintah '{cmd}' tidak didukung untuk penjadwalan.", "action": {"action": "error"}}
            
    return {"reply": "Aksi tidak dikenal.", "action": {"action": "error"}}

def get_music_duration(filepath: str) -> Optional[float]:
    try:
        import subprocess
        res = subprocess.run(
            ["ffprobe", "-i", filepath, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            capture_output=True, text=True, timeout=2.0
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass
    return None

_music_duration_cache = {}  # filepath -> (mtime, duration_str)

def get_cached_music_details() -> str:
    global _music_duration_cache
    if not os.path.exists(MUSIC_DIR):
        return "tidak ada lagu"
    
    try:
        files = sorted([f for f in os.listdir(MUSIC_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_MUSIC_EXTS])
    except Exception:
        return "tidak ada lagu"
        
    details = []
    for f in files:
        filepath = os.path.join(MUSIC_DIR, f)
        try:
            mtime = os.path.getmtime(filepath)
            # Check cache
            if filepath in _music_duration_cache and _music_duration_cache[filepath][0] == mtime:
                duration_str = _music_duration_cache[filepath][1]
            else:
                dur = get_music_duration(filepath)
                if dur:
                    mins = int(dur // 60)
                    secs = int(dur % 60)
                    duration_str = f"{mins}:{secs:02d}"
                else:
                    duration_str = "tidak diketahui"
                _music_duration_cache[filepath] = (mtime, duration_str)
            details.append(f"{f} (durasi {duration_str})")
        except Exception:
            details.append(f)
            
    return ", ".join(details) if details else "tidak ada lagu"

def get_context_prompts() -> str:
    now = datetime.datetime.now()
    months = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    
    music_str = get_cached_music_details()
    
    return (
        f"\n\n[SISTEM CONTEXT WAKTU]\nHari: {DAYS_ID[now.weekday()].title()}\n"
        f"Tanggal: {now.day} {months[now.month]} {now.year}\n"
        f"Waktu Sekarang: {now.strftime('%H:%M')}\n"
        f"Volume Default Sistem: {settings.get('default_volume', 50)}%\n"
        f"Volume Speaker Saat Ini: {get_system_volume()}%\n"
        f"Daftar File Musik Lokal Tersedia: {music_str}"
    )

def is_connected_to_internet() -> bool:
    import socket
    try:
        socket.setdefaulttimeout(1.5)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False

async def call_gemini_text(user_text: str, settings: dict) -> dict:
    if not is_connected_to_internet():
        return {"error": "Asisten gagal terhubung karena tidak ada koneksi internet. Pastikan WiFi Anda tersambung ke jaringan.", "error_code": "NO_INTERNET"}

    api_keys = settings.get("api_keys", [])
    active_model = settings.get("active_model", "gemini-1.5-flash")
    system_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not api_keys:
        return {"error": "API Key Gemini belum disetel di pengaturan.", "error_code": "NO_KEY"}
        
    # Compile prompt with date context
    prompt = system_prompt + get_context_prompts()
    payload = {
        "contents": list(CHAT_HISTORY) + [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": prompt}]},
        "tools": ALL_TOOLS
    }
    
    key_errors = []
    for idx, key in enumerate(api_keys):
        label = f"Key #{idx + 1}"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={key}"
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post(api_url, json=payload)
            if r.status_code == 200:
                res = r.json()
                parts = res.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                if parts:
                    part = parts[0]
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        return {"type": "function_call", "name": fc["name"], "args": fc.get("args", {})}
                    if "text" in part:
                        return {"type": "text", "text": part["text"]}
                key_errors.append({"key": label, "info": "Respons kosong"})
            else:
                key_errors.append({"key": label, "info": f"HTTP {r.status_code}: {r.text[:100]}"})
        except Exception as e:
            key_errors.append({"key": label, "info": str(e)})
            
    is_quota_error = any("429" in err["info"] for err in key_errors)
    is_key_error = any("400" in err["info"] for err in key_errors)
    
    if is_quota_error:
        err_msg = "Gagal menghubungi Gemini karena batas kuota harian (Rate Limit) API Key Anda sudah habis."
    elif is_key_error:
        err_msg = "API Key Gemini Anda tidak valid. Silakan periksa kembali di menu Pengaturan."
    else:
        err_msg = "Semua API Key gagal terhubung ke server Gemini. Silakan periksa jaringan internet Anda."
        
    return {"error": err_msg, "error_code": "ALL_KEYS_FAILED", "key_errors": key_errors}

# --- FastAPI ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# PWA service files
@app.get("/manifest.json")
async def get_manifest():
    return FileResponse(os.path.join(BASE_DIR, "static", "manifest.json"))

@app.get("/sw.js")
async def get_sw():
    return FileResponse(os.path.join(BASE_DIR, "static", "sw.js"), media_type="application/javascript")

@app.get("/icon-192.png")
async def get_icon_192():
    return FileResponse(os.path.join(BASE_DIR, "static", "icon-192.png"), media_type="image/png")

@app.get("/icon-512.png")
async def get_icon_512():
    return FileResponse(os.path.join(BASE_DIR, "static", "icon-512.png"), media_type="image/png")

# CCTV Stream
@app.get("/stream")
async def video_stream(request: Request):
    global active_camera_clients
    active_camera_clients += 1
    
    async def event_generator():
        global active_camera_clients, camera_active, latest_frame
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                if camera_active:
                    if latest_frame is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
                    else:
                        placeholder = generate_placeholder("LOADING CAMERA...")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                    await asyncio.sleep(0.04)  # ~25 FPS
                else:
                    placeholder = generate_placeholder("CCTV NONAKTIF")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                    await asyncio.sleep(0.5)
        finally:
            active_camera_clients = max(0, active_camera_clients - 1)
            settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
            if settings.get("camera_auto_off", True) and active_camera_clients <= 0:
                async def delayed_auto_off():
                    await asyncio.sleep(5)  # 5-second grace period for refresh
                    if active_camera_clients <= 0:
                        global camera_active
                        with camera_lock:
                            camera_active = False
                        print("No active CCTV stream clients after grace period. Auto-off camera triggered.")
                
                asyncio.create_task(delayed_auto_off())
                
    return StreamingResponse(event_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/ping")
async def api_ping():
    return {"ping": "pong"}

@app.get("/api/camera/state")
async def get_camera_state():
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    cctv_motion = settings.get("cctv_motion_detection_enabled", False)
    wifi_sensing = settings.get("wifi_sensing_enabled", False)
    wifi_method = settings.get("wifi_sensing_method", "ping")
    camera_sensing_active = wifi_sensing and (wifi_method == "camera")
    
    return {
        "active": camera_active,
        "motion_detection_enabled": cctv_motion,
        "camera_sensing_enabled": camera_sensing_active
    }

@app.post("/api/camera/toggle")
async def toggle_camera():
    global camera_active
    if not camera_active:
        camera_on()
    else:
        camera_off()
        
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    cctv_motion = settings.get("cctv_motion_detection_enabled", False)
    wifi_sensing = settings.get("wifi_sensing_enabled", False)
    wifi_method = settings.get("wifi_sensing_method", "ping")
    camera_sensing_active = wifi_sensing and (wifi_method == "camera")
    
    return {
        "active": camera_active,
        "motion_detection_enabled": cctv_motion,
        "camera_sensing_enabled": camera_sensing_active
    }


@app.post("/api/cctv/capture-baseline")
async def capture_baseline():
    global cap
    temp_cap = None
    ret = False
    frame = None
    try:
        if cap is None:
            temp_cap = cv2.VideoCapture(0)
            if not temp_cap.isOpened():
                return {"status": "error", "message": "Gagal membuka kamera."}
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(5):
                temp_cap.read()
                time.sleep(0.1)
            ret, frame = temp_cap.read()
        else:
            ret, frame = cap.read()
            
        if not ret or frame is None:
            if temp_cap:
                temp_cap.release()
            return {"status": "error", "message": "Gagal menangkap frame dari kamera."}
            
        os.makedirs(DATA_DIR, exist_ok=True)
        preview_path = os.path.join(DATA_DIR, "baseline_empty_preview.jpg")
        cv2.imwrite(preview_path, frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        calc_path = os.path.join(DATA_DIR, "baseline_empty_calc.png")
        cv2.imwrite(calc_path, gray_blurred)
        
        if temp_cap:
            temp_cap.release()
            
        add_log("CCTV_BASELINE_UPDATED", "CCTV", "Foto acuan kamar kosong (baseline) diperbarui.")
        return {"status": "success", "message": "Foto baseline berhasil diambil!"}
    except Exception as e:
        if temp_cap:
            temp_cap.release()
        return {"status": "error", "message": f"Error: {e}"}

@app.post("/api/cctv/delete-baseline")
async def delete_baseline():
    preview_path = os.path.join(DATA_DIR, "baseline_empty_preview.jpg")
    calc_path = os.path.join(DATA_DIR, "baseline_empty_calc.png")
    try:
        if os.path.exists(preview_path):
            os.remove(preview_path)
        if os.path.exists(calc_path):
            os.remove(calc_path)
        add_log("CCTV_BASELINE_DELETED", "CCTV", "Foto acuan kamar kosong (baseline) dihapus.")
        return {"status": "success", "message": "Foto baseline berhasil dihapus."}
    except Exception as e:
        return {"status": "error", "message": f"Gagal menghapus baseline: {e}"}

@app.get("/api/cctv/baseline-preview")
async def get_baseline_preview():
    preview_path = os.path.join(DATA_DIR, "baseline_empty_preview.jpg")
    if os.path.exists(preview_path):
        return FileResponse(preview_path)
    return JSONResponse(status_code=404, content={"status": "error", "message": "Belum ada foto baseline."})


@app.post("/api/camera/record")
async def api_record_video(request: Request, background_tasks: BackgroundTasks):
    global is_recording_video
    if is_recording_video:
        return {"status": "busy", "message": "Rekaman sedang berjalan."}
    data = await request.json()
    duration = max(1, min(60, int(data.get("duration", 5))))
    
    result = {"filename": None}
    def do_record():
        fn = record_video(duration_sec=duration)
        result["filename"] = fn
    
    import concurrent.futures
    loop = asyncio.get_event_loop()
    filename = await loop.run_in_executor(None, lambda: record_video(duration_sec=duration))
    if filename:
        return {"status": "success", "filename": filename, "url": f"/static/videos/{filename}", "duration": duration}
    else:
        return {"status": "error", "message": "Gagal merekam video."}


@app.get("/api/camera/snapshots")
async def get_snapshots():
    snapshots = []
    if os.path.exists(SNAPSHOTS_DIR):
        for f in sorted(os.listdir(SNAPSHOTS_DIR), reverse=True):
            if f.startswith("snap_") and f.endswith(".jpg"):
                path = os.path.join(SNAPSHOTS_DIR, f)
                stat = os.stat(path)
                try:
                    parts = f.split("_")
                    date_part = parts[1]
                    time_part = parts[2].split(".")[0]
                    display_time = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                except Exception:
                    display_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
                snapshots.append({
                    "filename": f,
                    "url": f"/static/snapshots/{f}",
                    "time": display_time,
                    "size": stat.st_size
                })
    return snapshots

@app.delete("/api/camera/snapshots/{filename}")
async def delete_snapshot(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    fpath = os.path.join(SNAPSHOTS_DIR, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="File not found")

# Alarm Endpoints
@app.get("/api/alarms")
async def api_get_alarms():
    return get_alarms()

@app.post("/api/alarms")
async def api_create_alarm(request: Request):
    data = await request.json()
    alarms = get_alarms()
    new_alarm = {
        "id": str(uuid.uuid4()),
        "name": data.get("name", "Alarm Baru"),
        "time": data.get("time", "07:00"),
        "type": data.get("type", "tts"),
        "tts_text": data.get("tts_text", "Selamat {salam}, saatnya bangun."),
        "tts_lang": data.get("tts_lang", "id-ID-GadisNeural"),
        "ringtone": data.get("ringtone", ""),
        "repeat_days": data.get("repeat_days", []),
        "volume": int(data.get("volume", 80)),
        "enabled": data.get("enabled", True),
        "snooze_duration": int(data.get("snooze_duration", 5)),
        "last_triggered": ""
    }
    alarms.append(new_alarm)
    save_alarms(alarms)
    add_log("CREATE", new_alarm["name"], f"Alarm dibuat untuk jam {new_alarm['time']}")
    return new_alarm

@app.put("/api/alarms/{alarm_id}")
async def api_update_alarm(alarm_id: str, request: Request):
    data = await request.json()
    alarms = get_alarms()
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        raise HTTPException(status_code=404, detail="Alarm tidak ditemukan")
        
    alarm["name"] = data.get("name", alarm["name"])
    alarm["time"] = data.get("time", alarm["time"])
    alarm["type"] = data.get("type", alarm["type"])
    alarm["tts_text"] = data.get("tts_text", alarm["tts_text"])
    alarm["tts_lang"] = data.get("tts_lang", alarm.get("tts_lang", "id-ID-GadisNeural"))
    alarm["ringtone"] = data.get("ringtone", alarm["ringtone"])
    alarm["repeat_days"] = data.get("repeat_days", alarm["repeat_days"])
    alarm["volume"] = int(data.get("volume", alarm["volume"]))
    alarm["enabled"] = data.get("enabled", alarm["enabled"])
    alarm["snooze_duration"] = int(data.get("snooze_duration", alarm["snooze_duration"]))
    
    save_alarms(alarms)
    add_log("UPDATE", alarm["name"], "Alarm diperbarui")
    return alarm

@app.delete("/api/alarms/{alarm_id}")
async def api_delete_alarm(alarm_id: str):
    alarms = get_alarms()
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        raise HTTPException(status_code=404, detail="Alarm tidak ditemukan")
    alarms = [a for a in alarms if a["id"] != alarm_id]
    save_alarms(alarms)
    snoozed_alarms.pop(alarm_id, None)
    add_log("DELETE", alarm["name"], "Alarm dihapus")
    return {"status": "success"}

@app.post("/api/alarms/{alarm_id}/toggle")
async def api_toggle_alarm(alarm_id: str):
    alarms = get_alarms()
    alarm = next((a for a in alarms if a["id"] == alarm_id), None)
    if not alarm:
        raise HTTPException(status_code=404, detail="Alarm tidak ditemukan")
    alarm["enabled"] = not alarm["enabled"]
    save_alarms(alarms)
    status_str = "diaktifkan" if alarm["enabled"] else "dimatikan"
    add_log("TOGGLE", alarm["name"], f"Alarm {status_str}")
    return alarm

@app.get("/api/active-alarm")
async def api_active_alarm():
    global is_alarm_playing, active_alarm_id, active_alarm_name, active_alarm_type, active_alarm_value, active_alarm_volume
    return {
        "is_playing": is_alarm_playing,
        "id": active_alarm_id,
        "name": active_alarm_name,
        "type": active_alarm_type,
        "value": active_alarm_value,
        "volume": active_alarm_volume
    }

def play_morning_briefing_task():
    try:
        # Give a small 2-second sleep to let PulseAudio finish closing the previous stream
        time.sleep(2.0)
        
        # Run get_briefing_text
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        text = loop.run_until_complete(get_briefing_text())
        greeting = f"Selamat pagi! Berikut adalah briefing harian Anda. {text}"
        
        briefing_file = os.path.join(UPLOAD_DIR, "briefing_temp.mp3")
        settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
        lang = settings.get("stt_lang", "id-ID")
        
        loop.run_until_complete(generate_tts_file_async(greeting, lang, briefing_file))
        vol = settings.get("default_volume", 50)
        play_audio_file(briefing_file, volume=vol, is_alarm=False)
    except Exception as e:
        print(f"Error playing morning briefing: {e}")

@app.post("/api/alarms/{alarm_id}/dismiss")
async def api_dismiss_alarm(alarm_id: str, background_tasks: BackgroundTasks):
    global active_alarm_name
    add_log("DISMISS", active_alarm_name or "Alarm", "Alarm dimatikan")
    stop_active_audio()
    
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    if settings.get("alarm_briefing_enabled", True):
        background_tasks.add_task(play_morning_briefing_task)
        
    return {"status": "success"}

@app.post("/api/alarms/{alarm_id}/snooze")
async def api_snooze_alarm(alarm_id: str):
    global active_alarm_id, active_alarm_name, snoozed_alarms
    target_id = alarm_id if (alarm_id != "null" and alarm_id != "") else active_alarm_id
    if target_id:
        alarms = get_alarms()
        alarm = next((a for a in alarms if a["id"] == target_id), None)
        duration = alarm["snooze_duration"] if alarm else 5
        
        trigger_epoch = time.time() + (duration * 60)
        snoozed_alarms[target_id] = trigger_epoch
        add_log("SNOOZE", active_alarm_name or "Alarm", f"Alarm ditunda selama {duration} menit")
    stop_active_audio()
    return {"status": "success"}

# Volume Status
@app.get("/api/audio/volume")
async def api_get_volume():
    return {"volume": get_system_volume(), "default_volume": get_configured_default_volume()}

@app.post("/api/audio/volume")
async def api_set_volume(level: int):
    set_system_volume(level)
    return {"status": "success", "volume": level}

def get_all_sinks() -> list:
    """Return list of all available audio sinks with friendly labels."""
    sinks = []
    try:
        res = subprocess.run(["pactl", "list", "sinks"], capture_output=True, text=True)
        default_res = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True)
        default_sink = default_res.stdout.strip() if default_res.returncode == 0 else ""

        current = {}
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("Sink #"):
                if current.get("name"):
                    sinks.append(current)
                current = {"id": line.split("#")[1]}
            elif line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
                current["name"] = name
                current["is_default"] = (name == default_sink)
                # Detect type
                if "bluez" in name.lower():
                    current["type"] = "bluetooth"
                elif "hdmi" in name.lower():
                    current["type"] = "hdmi"
                elif "usb" in name.lower():
                    current["type"] = "usb"
                else:
                    current["type"] = "builtin"
            elif line.startswith("Description:"):
                current["label"] = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                current["state"] = line.split(":", 1)[1].strip()
        if current.get("name"):
            sinks.append(current)
    except Exception as e:
        print(f"get_all_sinks error: {e}")
    return sinks

@app.get("/api/audio/sinks")
async def api_list_sinks():
    """List all available audio output devices (sinks)."""
    return get_all_sinks()

@app.post("/api/audio/sink/set")
async def api_set_sink(request: Request):
    """Switch the active audio output device."""
    data = await request.json()
    sink_name = data.get("sink_name", "").strip()
    if not sink_name:
        return JSONResponse(status_code=400, content={"detail": "sink_name required"})
    try:
        # Set as default sink
        subprocess.run(["pactl", "set-default-sink", sink_name], capture_output=True)
        # Move all current streams to this sink
        inputs_res = subprocess.run(["pactl", "list", "sink-inputs", "short"], capture_output=True, text=True)
        if inputs_res.returncode == 0:
            for line in inputs_res.stdout.splitlines():
                if line.strip():
                    input_id = line.split()[0]
                    subprocess.run(["pactl", "move-sink-input", input_id, sink_name], capture_output=True)
        # Restore volume on new sink
        default_vol = get_configured_default_volume()
        subprocess.run(["pactl", "set-sink-volume", sink_name, f"{default_vol}%"], capture_output=True)
        subprocess.run(["pactl", "set-sink-mute", sink_name, "0"], capture_output=True)
        add_log("AUDIO_SINK_SWITCH", "Audio", f"Output audio diganti ke: {sink_name}")
        return {"status": "success", "sink_name": sink_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Ringtone Uploads & List
@app.get("/api/audio/list")
async def api_list_audio():
    audio_files = []
    for f in sorted(os.listdir(UPLOAD_DIR)):
        if (f.endswith(".mp3") or f.endswith(".wav") or f.endswith(".m4a") or f.endswith(".webm")) and not f.startswith("tts_") and not f.endswith("_play.wav"):
            path = os.path.join(UPLOAD_DIR, f)
            stat = os.stat(path)
            audio_files.append({
                "filename": f,
                "size": stat.st_size
            })
    return audio_files

@app.post("/api/audio/upload")
async def api_upload_audio(file: UploadFile = File(...)):
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-").strip()
    if not safe_filename:
        safe_filename = "ringtone.mp3"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger WAV conversion
    wav_path = file_path.rsplit(".", 1)[0] + "_play.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True, text=True
        )
    except Exception:
        pass
    return {"status": "success", "filename": safe_filename}

@app.post("/api/audio/generate_tts_ringtone")
async def api_generate_tts_ringtone(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "default")
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")
        
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    tts_engine = settings.get("tts_engine", "edge-tts")
    voice_setting = settings.get("tts_voice", "female")
    
    timestamp = int(time.time())
    is_piper = (lang == "default" and tts_engine == "piper")
    ext = ".wav" if is_piper else ".mp3"
    
    safe_text = "".join(c for c in text[:15] if c.isalnum() or c in " -").strip().replace(" ", "_")
    filename = f"tts-ringtone-{timestamp}-{safe_text}{ext}" if safe_text else f"tts-ringtone-{timestamp}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        if is_piper:
            # Run Piper directly
            piper_bin = "/app/bin/piper/piper"
            model_path = "/app/models/id_ID-news_tts-medium.onnx"
            if not os.path.exists(piper_bin):
                piper_bin = os.path.join(os.path.dirname(__file__), "bin", "piper", "piper")
                model_path = os.path.join(os.path.dirname(__file__), "models", "id_ID-news_tts-medium.onnx")
                
            if os.path.exists(piper_bin) and os.path.exists(model_path):
                temp_wav = os.path.join(UPLOAD_DIR, f"ring_temp_{timestamp}.wav")
                proc = await asyncio.create_subprocess_exec(
                    piper_bin, "--model", model_path, "--output_file", temp_wav,
                    "--length_scale", "1.12",
                    "--sentence_silence", "0.35",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate(input=text.encode('utf-8'))
                
                if os.path.exists(temp_wav):
                    if voice_setting == "male":
                        # Pitch shift using rubberband for high quality
                        proc_ffmpeg = await asyncio.create_subprocess_exec(
                            "ffmpeg", "-y", "-i", temp_wav, "-af", "rubberband=pitch=0.82", file_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await proc_ffmpeg.communicate()
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    else:
                        os.rename(temp_wav, file_path)
                else:
                    raise Exception("Piper failed to output wave file")
            else:
                # Fallback to Edge-TTS
                await generate_tts_file_async(text, "id-ID-GadisNeural" if voice_setting == "female" else "id-ID-ArdiNeural", file_path)
        else:
            tts_voice = lang
            if lang == "default":
                tts_voice = "id-ID-GadisNeural" if voice_setting == "female" else "id-ID-ArdiNeural"
            await generate_tts_file_async(text, tts_voice, file_path)
            
        # Trigger WAV conversion for playback compat
        wav_path = file_path.rsplit(".", 1)[0] + "_play.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", wav_path],
                capture_output=True, text=True
            )
        except Exception:
            pass
        add_log("CREATE", "TTS Ringtone", f"Membuat ringtone kustom TTS: '{text[:25]}'")
        return {"status": "success", "filename": filename, "timestamp": timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/audio/delete/{filename}")
async def api_delete_audio(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="File name invalid")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + "_play.wav"
    if os.path.exists(wav_path):
        os.remove(wav_path)
    return {"status": "success"}

# Settings & Preferences
@app.get("/api/settings")
async def get_settings():
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    for k, v in DEFAULT_SETTINGS.items():
        if k not in settings:
            settings[k] = v
    settings["volume"] = get_system_volume()
    settings["default_volume"] = get_configured_default_volume()
    return settings

@app.post("/api/settings")
async def post_settings(request: Request):
    data = await request.json()
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    for k, v in DEFAULT_SETTINGS.items():
        if k not in settings:
            settings[k] = v

    if "api_keys" in data:
        settings["api_keys"] = [k.strip() for k in data["api_keys"] if k.strip()]
    if "system_prompt" in data:
        settings["system_prompt"] = data["system_prompt"].strip()
    if "active_model" in data:
        settings["active_model"] = data["active_model"].strip()
    if "stt_lang" in data:
        settings["stt_lang"] = data["stt_lang"].strip()
    if "tts_voice" in data:
        settings["tts_voice"] = data["tts_voice"].strip()
    if "volume" in data:
        vol = int(data["volume"])
        settings["volume"] = vol
        set_system_volume(vol)
    if "default_volume" in data:
        settings["default_volume"] = int(data["default_volume"])
    if "response_mode" in data:
        settings["response_mode"] = data["response_mode"]
    if "cctv_snapshot_enabled" in data:
        settings["cctv_snapshot_enabled"] = bool(data["cctv_snapshot_enabled"])
    if "cctv_snapshot_interval" in data:
        settings["cctv_snapshot_interval"] = max(1, int(data["cctv_snapshot_interval"]))
    if "alarm_briefing_enabled" in data:
        settings["alarm_briefing_enabled"] = bool(data["alarm_briefing_enabled"])
    if "keepalive_interval_min" in data:
        settings["keepalive_interval_min"] = max(0, int(data["keepalive_interval_min"]))
    if "camera_auto_off" in data:
        settings["camera_auto_off"] = bool(data["camera_auto_off"])
    if "immich_api_key" in data:
        settings["immich_api_key"] = data["immich_api_key"].strip()
    if "immich_address" in data:
        settings["immich_address"] = data["immich_address"].strip()
    if "immich_sync_enabled" in data:
        settings["immich_sync_enabled"] = bool(data["immich_sync_enabled"])
    if "wifi_sensing_enabled" in data:
        settings["wifi_sensing_enabled"] = bool(data["wifi_sensing_enabled"])
    if "wifi_sensing_method" in data:
        settings["wifi_sensing_method"] = data["wifi_sensing_method"].strip()
    if "wifi_sensing_target_ip" in data:
        settings["wifi_sensing_target_ip"] = data["wifi_sensing_target_ip"].strip()
    if "wifi_sensing_target_mac" in data:
        settings["wifi_sensing_target_mac"] = data["wifi_sensing_target_mac"].strip()
    if "cctv_motion_detection_enabled" in data:
        settings["cctv_motion_detection_enabled"] = bool(data["cctv_motion_detection_enabled"])
    if "presence_stillness_threshold_min" in data:
        settings["presence_stillness_threshold_min"] = float(data["presence_stillness_threshold_min"])
    if "presence_short_limit_min" in data:
        settings["presence_short_limit_min"] = float(data["presence_short_limit_min"])
    if "presence_medium_limit_min" in data:
        settings["presence_medium_limit_min"] = float(data["presence_medium_limit_min"])
    if "presence_long_limit_min" in data:
        settings["presence_long_limit_min"] = float(data["presence_long_limit_min"])
    if "cctv_fps_day" in data:
        settings["cctv_fps_day"] = max(1, int(data["cctv_fps_day"]))
    if "cctv_fps_night" in data:
        settings["cctv_fps_night"] = max(1, int(data["cctv_fps_night"]))
    if "cctv_fps_night_start_hour" in data:
        settings["cctv_fps_night_start_hour"] = min(23, max(0, int(data["cctv_fps_night_start_hour"])))
    if "cctv_fps_night_end_hour" in data:
        settings["cctv_fps_night_end_hour"] = min(23, max(0, int(data["cctv_fps_night_end_hour"])))
    if "tts_engine" in data:
        settings["tts_engine"] = data["tts_engine"].strip()
    if "stt_engine" in data:
        settings["stt_engine"] = data["stt_engine"].strip()

    write_json_file(SETTINGS_FILE, settings)
    return {"status": "success", "settings": settings}

# Logs & History
@app.get("/api/logs")
async def api_get_logs():
    return read_json_file(LOGS_FILE, [])

# Music Endpoints
@app.get("/api/music/list")
async def list_music():
    try:
        files = []
        for fname in sorted(os.listdir(MUSIC_DIR)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_MUSIC_EXTS:
                fpath = os.path.join(MUSIC_DIR, fname)
                files.append({
                    "filename": fname,
                    "size": os.path.getsize(fpath),
                    "playing": current_playing_name == fname
                })
        return {"status": "success", "files": files, "current_playing": current_playing_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/music/play/{filename}")
async def play_music_file(filename: str):
    safe_name = os.path.basename(filename)
    result = play_local_music(safe_name)
    return {"status": "success", "message": result, "playing": current_playing_name}

@app.post("/api/music/play_stream")
async def api_play_music_stream(request: Request):
    data = await request.json()
    genre = data.get("genre", "lofi")
    result = play_music_stream(genre)
    return {"status": "success", "message": result, "playing": current_playing_name}

@app.post("/api/music/play_youtube")
async def api_play_youtube(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query empty")
    result = play_youtube_audio(query)
    return {"status": "success", "message": result, "playing": current_playing_name}

@app.post("/api/music/stop")
async def api_stop_music():
    result = stop_music_play()
    return {"status": "success", "message": result}

@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_MUSIC_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
    safe_name = re.sub(r"[^\w\-_. ]", "_", file.filename or "upload").strip()
    dest_path = os.path.join(MUSIC_DIR, safe_name)
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        return {"status": "success", "filename": safe_name, "size": len(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/music/{filename}")
async def delete_music(filename: str):
    safe_name = os.path.basename(filename)
    fpath = os.path.join(MUSIC_DIR, safe_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(fpath)
        return {"status": "success", "deleted": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/music/rename")
async def rename_music(request: Request):
    data = await request.json()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    
    if not old_name or not new_name:
        raise HTTPException(status_code=400, detail="Nama file lama/baru tidak boleh kosong")
        
    if "/" in old_name or "\\" in old_name or "/" in new_name or "\\" in new_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    old_ext = os.path.splitext(old_name)[1].lower()
    new_ext = os.path.splitext(new_name)[1].lower()
    if old_ext != new_ext:
        new_name = os.path.splitext(new_name)[0] + old_ext
        
    old_path = os.path.join(MUSIC_DIR, old_name)
    new_path = os.path.join(MUSIC_DIR, new_name)
    
    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="File lama tidak ditemukan")
    if os.path.exists(new_path) and old_name != new_name:
        raise HTTPException(status_code=400, detail="Nama file baru sudah digunakan")
        
    try:
        os.rename(old_path, new_path)
        old_wav = old_path.rsplit(".", 1)[0] + "_play.wav"
        new_wav = new_path.rsplit(".", 1)[0] + "_play.wav"
        if os.path.exists(old_wav):
            os.rename(old_wav, new_wav)
        return {"status": "success", "new_name": new_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Voice Assistant Endpoints

@app.post("/api/stt/whisper")
async def api_stt_whisper(file: UploadFile = File(...)):
    """Transcribe uploaded audio file using local whisper-cli."""
    WHISPER_BIN = "/app/bin/whisper/whisper-cli"
    WHISPER_MODEL = "/app/models/ggml-tiny.bin"
    WHISPER_LIBS = "/app/bin/whisper"
    
    if not os.path.exists(WHISPER_BIN):
        # Try host path fallback
        WHISPER_BIN = os.path.join(os.path.dirname(__file__), "bin", "whisper", "whisper-cli")
        WHISPER_MODEL = os.path.join(os.path.dirname(__file__), "models", "ggml-tiny.bin")
        WHISPER_LIBS = os.path.join(os.path.dirname(__file__), "bin", "whisper")

    if not os.path.exists(WHISPER_BIN):
        raise HTTPException(status_code=501, detail="Whisper-cli not installed")
    if not os.path.exists(WHISPER_MODEL):
        raise HTTPException(status_code=501, detail="Whisper model not found")

    ts = int(time.time() * 1000)
    ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    raw_path = os.path.join(UPLOAD_DIR, f"stt_{ts}{ext}")
    wav_path = os.path.join(UPLOAD_DIR, f"stt_{ts}.wav")
    
    try:
        with open(raw_path, "wb") as f:
            f.write(await file.read())
        
        # Convert to 16kHz mono WAV for whisper
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="Failed to convert audio")
        
        # Run whisper-cli
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = WHISPER_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
        
        proc = await asyncio.create_subprocess_exec(
            WHISPER_BIN,
            "--model", WHISPER_MODEL,
            "--file", wav_path,
            "--language", "id",
            "--output-txt",
            "--no-prints",
            "--threads", "2",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        stdout, stderr = await proc.communicate()
        
        # Whisper outputs text to <file>.txt
        txt_path = wav_path + ".txt"
        transcript = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            os.remove(txt_path)
        elif stdout:
            transcript = stdout.decode("utf-8", errors="ignore").strip()
        
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [raw_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/api/chat")
async def chat(background_tasks: BackgroundTasks, request: Request):
    data = await request.json()
    user_text = data.get("text", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Text empty")
        
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    
    # Try local router first
    local_res = await local_intent_router(user_text)
    if local_res:
        reply_text = local_res["reply"]
        action_info = local_res["action"]
    else:
        if not settings.get("api_keys"):
            raise HTTPException(status_code=400, detail="Gemini API Key empty and no local command matched.")
            
        result = await call_gemini_text(user_text, settings)
        if "error" in result:
            # Fallback to local if Gemini fails
            raise HTTPException(status_code=502, detail=result["error"])
        elif result.get("type") == "function_call":
            handled = await execute_function_call(result["name"], result.get("args", {}))
            reply_text = handled["reply"]
            action_info = handled["action"]
        else:
            reply_text = result["text"]
            action_info = {"action": "none"}
            
    # Save to history
    global CHAT_HISTORY
    CHAT_HISTORY.append({"role": "user", "parts": [{"text": user_text}]})
    CHAT_HISTORY.append({"role": "model", "parts": [{"text": reply_text}]})
    CHAT_HISTORY = CHAT_HISTORY[-12:] # limit history length
    
    if reply_text:
        background_tasks.add_task(speak, reply_text)
        
    return {"transcript": user_text, "reply": reply_text, "action": action_info}

@app.post("/api/chat/clear")
async def clear_chat():
    global CHAT_HISTORY
    CHAT_HISTORY.clear()
    return {"status": "success", "message": "History cleared"}

@app.post("/api/audio/stop")
async def stop_all_audio():
    stop_active_audio()
    stop_music_play()
    return {"status": "success"}

# --- BLUETOOTH API ENDPOINTS ---

@app.get("/api/bluetooth/status")
async def api_bluetooth_status():
    ctrl = get_bluetooth_controller_status()
    devices = get_bluetooth_devices()
    bt_cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    
    return {
        "controller": ctrl,
        "scanning": bluetooth_scanning,
        "paired": devices["paired"],
        "connected": devices["connected"],
        "all_discovered": devices["all"],
        "auto_reconnect_mac": bt_cfg.get("auto_reconnect_mac"),
        "auto_reconnect_enabled": bt_cfg.get("auto_reconnect_enabled")
    }

@app.post("/api/bluetooth/power")
async def api_bluetooth_power(request: Request):
    data = await request.json()
    state = "on" if data.get("power", True) else "off"
    res = subprocess.run(["bluetoothctl", "power", state], capture_output=True, text=True)
    return {"status": "success", "output": res.stdout.strip()}

@app.post("/api/bluetooth/scan")
async def api_bluetooth_scan(background_tasks: BackgroundTasks):
    global bluetooth_scanning
    if bluetooth_scanning:
        return {"status": "scanning", "message": "Scan already running"}
    background_tasks.add_task(run_bluetooth_scan)
    return {"status": "success", "message": "Scan started in background"}

@app.post("/api/bluetooth/connect")
async def api_bluetooth_connect(request: Request):
    data = await request.json()
    mac = data.get("mac")
    if not mac:
        raise HTTPException(status_code=400, detail="MAC is required")
    
    # Run connect sequence
    add_log("BLUETOOTH_CONNECT", "Bluetooth", f"Menghubungkan ke {mac}...")
    # Attempt to pair first (some devices require pairing first to register properly)
    subprocess.run(["bluetoothctl", "pair", mac], capture_output=True, timeout=15.0)
    subprocess.run(["bluetoothctl", "trust", mac], capture_output=True)
    res = subprocess.run(["bluetoothctl", "connect", mac], capture_output=True, text=True)
    
    success = "Connection successful" in res.stdout or "successful" in res.stdout.lower()
    if success:
        add_log("BLUETOOTH_SUCCESS", "Bluetooth", f"Berhasil terhubung ke {mac}")
        return {"status": "success", "output": res.stdout.strip()}
    else:
        add_log("BLUETOOTH_FAILED", "Bluetooth", f"Gagal terhubung ke {mac}: {res.stderr or res.stdout}")
        return {"status": "failed", "output": res.stdout.strip() + "\n" + res.stderr}

@app.post("/api/bluetooth/disconnect")
async def api_bluetooth_disconnect(request: Request):
    data = await request.json()
    mac = data.get("mac")
    if not mac:
        raise HTTPException(status_code=400, detail="MAC is required")
        
    res = subprocess.run(["bluetoothctl", "disconnect", mac], capture_output=True, text=True)
    add_log("BLUETOOTH_DISCONNECT", "Bluetooth", f"Memutuskan hubungan dari {mac}")
    return {"status": "success", "output": res.stdout.strip()}

@app.post("/api/bluetooth/pair")
async def api_bluetooth_pair(request: Request):
    data = await request.json()
    mac = data.get("mac")
    if not mac:
        raise HTTPException(status_code=400, detail="MAC is required")
        
    res = subprocess.run(["bluetoothctl", "pair", mac], capture_output=True, text=True)
    return {"status": "success", "output": res.stdout.strip()}

@app.post("/api/bluetooth/unpair")
async def api_bluetooth_unpair(request: Request):
    data = await request.json()
    mac = data.get("mac")
    if not mac:
        raise HTTPException(status_code=400, detail="MAC is required")
        
    res = subprocess.run(["bluetoothctl", "remove", mac], capture_output=True, text=True)
    return {"status": "success", "output": res.stdout.strip()}

@app.get("/api/bluetooth/settings")
async def api_bluetooth_get_settings():
    cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    return cfg

@app.post("/api/bluetooth/settings")
async def api_bluetooth_save_settings(request: Request):
    data = await request.json()
    cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    if "auto_reconnect_mac" in data:
        cfg["auto_reconnect_mac"] = data["auto_reconnect_mac"]
    if "auto_reconnect_enabled" in data:
        cfg["auto_reconnect_enabled"] = bool(data["auto_reconnect_enabled"])
    if "auto_switch_to_bt" in data:
        cfg["auto_switch_to_bt"] = bool(data["auto_switch_to_bt"])
    write_json_file(BLUETOOTH_FILE, cfg)
    return {"status": "success", "config": cfg}

@app.post("/api/bluetooth/auto-reconnect")
async def api_bluetooth_auto_reconnect(request: Request):
    data = await request.json()
    mac = data.get("mac")
    enabled = bool(data.get("enabled", False))
    auto_switch = bool(data.get("auto_switch_to_bt", True))
    
    bt_cfg = read_json_file(BLUETOOTH_FILE, DEFAULT_BLUETOOTH_SETTINGS)
    bt_cfg["auto_reconnect_mac"] = mac
    bt_cfg["auto_reconnect_enabled"] = enabled
    bt_cfg["auto_switch_to_bt"] = auto_switch
    write_json_file(BLUETOOTH_FILE, bt_cfg)
    return {"status": "success", "config": bt_cfg}

# --- VOICE NOTE (VN) / INTERCOM ENDPOINTS ---
@app.get("/api/audio/vn_list")
async def api_list_vn():
    vns = []
    if os.path.exists(UPLOAD_DIR):
        for f in sorted(os.listdir(UPLOAD_DIR)):
            if (f.startswith("vn_") or f.startswith("upload_")) and not f.endswith(".wav") and not f.endswith("_play.wav"):
                path = os.path.join(UPLOAD_DIR, f)
                stat = os.stat(path)
                try:
                    parts = f.split("_")
                    timestamp = int(parts[1])
                except Exception:
                    timestamp = int(os.path.getmtime(path))
                    
                if f.startswith("vn_"):
                    display_name = "Voice Note"
                else:
                    parts = f.split("_", 2)
                    display_name = parts[2] if len(parts) > 2 else f
                    if display_name.startswith("TTS_"):
                        display_name = "TTS: " + display_name[4:].replace("_", " ").rsplit(".", 1)[0]
                    
                vns.append({
                    "filename": f,
                    "display_name": display_name,
                    "url": f"/uploads/{f}",
                    "timestamp": timestamp,
                    "size": stat.st_size
                })
    return sorted(vns, key=lambda x: x["timestamp"], reverse=True)

@app.post("/api/audio/vn_upload")
async def api_vn_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    timestamp = int(time.time())
    file_name = f"vn_{timestamp}.webm"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(play_audio_file, file_path)
    return {"status": "success", "filename": file_name, "timestamp": timestamp}

@app.post("/api/audio/vn_upload_file")
async def api_vn_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    timestamp = int(time.time())
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-").strip()
    if not safe_filename:
        safe_filename = "audio.mp3"
        
    file_name = f"upload_{timestamp}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(play_audio_file, file_path)
    return {"status": "success", "filename": file_name, "timestamp": timestamp}

@app.post("/api/audio/vn_tts")
async def api_vn_tts(background_tasks: BackgroundTasks, request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "default")
    if not text:
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")
        
    settings = read_json_file(SETTINGS_FILE, DEFAULT_SETTINGS)
    tts_engine = settings.get("tts_engine", "edge-tts")
    voice_setting = settings.get("tts_voice", "female")
    
    timestamp = int(time.time())
    is_piper = (lang == "default" and tts_engine == "piper")
    ext = ".wav" if is_piper else ".mp3"
    
    safe_text = "".join(c for c in text[:15] if c.isalnum() or c in " -").strip().replace(" ", "_")
    file_name = f"upload_{timestamp}_TTS_{safe_text}{ext}" if safe_text else f"upload_{timestamp}_TTS{ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    try:
        if is_piper:
            # Run Piper directly
            piper_bin = "/app/bin/piper/piper"
            model_path = "/app/models/id_ID-news_tts-medium.onnx"
            if not os.path.exists(piper_bin):
                piper_bin = os.path.join(os.path.dirname(__file__), "bin", "piper", "piper")
                model_path = os.path.join(os.path.dirname(__file__), "models", "id_ID-news_tts-medium.onnx")
                
            if os.path.exists(piper_bin) and os.path.exists(model_path):
                temp_wav = os.path.join(UPLOAD_DIR, f"vn_temp_{timestamp}.wav")
                proc = await asyncio.create_subprocess_exec(
                    piper_bin, "--model", model_path, "--output_file", temp_wav,
                    "--length_scale", "1.12",
                    "--sentence_silence", "0.35",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate(input=text.encode('utf-8'))
                
                if os.path.exists(temp_wav):
                    if voice_setting == "male":
                        # Pitch shift using rubberband for high quality
                        proc_ffmpeg = await asyncio.create_subprocess_exec(
                            "ffmpeg", "-y", "-i", temp_wav, "-af", "rubberband=pitch=0.82", file_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await proc_ffmpeg.communicate()
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                    else:
                        os.rename(temp_wav, file_path)
                else:
                    raise Exception("Piper failed to output wave file")
            else:
                # Fallback to Edge-TTS
                await generate_tts_file_async(text, "id-ID-GadisNeural" if voice_setting == "female" else "id-ID-ArdiNeural", file_path)
        else:
            tts_voice = lang
            if lang == "default":
                tts_voice = "id-ID-GadisNeural" if voice_setting == "female" else "id-ID-ArdiNeural"
            await generate_tts_file_async(text, tts_voice, file_path)
            
        background_tasks.add_task(play_audio_file, file_path)
        return {"status": "success", "filename": file_name, "timestamp": timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/vn_play/{filename}")
async def api_vn_play(filename: str, background_tasks: BackgroundTasks):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    background_tasks.add_task(play_audio_file, file_path)
    return {"status": "success"}

@app.delete("/api/audio/vn_delete/{filename}")
async def api_vn_delete(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    wav_path = file_path.rsplit(".", 1)[0] + "_play.wav"
    if os.path.exists(wav_path):
        os.remove(wav_path)
        
    if filename.startswith("vn_"):
        legacy_wav = os.path.join(UPLOAD_DIR, filename.replace(".webm", ".wav"))
        if os.path.exists(legacy_wav):
            os.remove(legacy_wav)
    elif filename.startswith("upload_"):
        parts = filename.split("_")
        if len(parts) >= 2:
            legacy_wav = os.path.join(UPLOAD_DIR, f"upload_{parts[1]}.wav")
            if os.path.exists(legacy_wav):
                os.remove(legacy_wav)
                
    return {"status": "success"}

# --- WIFI HELPER FUNCTIONS ---
def get_wifi_status() -> dict:
    try:
        # Run nmcli to check connection status
        res = subprocess.run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"], capture_output=True, text=True, timeout=5.0)
        connected_ssid = None
        state = "disconnected"
        interface = "wlo1"
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if line.startswith("Warning"):
                    continue
                parts = line.split(":")
                if len(parts) >= 4 and parts[1] == "wifi":
                    interface = parts[0]
                    state = parts[2]
                    if parts[2] in ["connected", "terhubung"]:
                        connected_ssid = parts[3]
                        
        if (state in ["connected", "terhubung"]) and connected_ssid:
            # Get details of the active connection (signal, bssid, security)
            details_res = subprocess.run(["nmcli", "-t", "-f", "active,ssid,signal,bssid,security", "device", "wifi"], capture_output=True, text=True, timeout=5.0)
            signal = "--"
            bssid = "--"
            security = "--"
            if details_res.returncode == 0:
                for line in details_res.stdout.splitlines():
                    if line.startswith("Warning"):
                        continue
                    if line.startswith("yes:") or line.startswith("ya:"):
                        parts = line.split(":")
                        if len(parts) >= 5:
                            signal = parts[2]
                            bssid = parts[3].replace("\\", "")
                            security = parts[4]
                            break
            
            # Get IP address
            ip = "--"
            ip_res = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3.0)
            if ip_res.returncode == 0 and ip_res.stdout.strip():
                ip = ip_res.stdout.split()[0]
                
            return {
                "connected": True,
                "ssid": connected_ssid,
                "interface": interface,
                "signal": signal,
                "bssid": bssid,
                "security": security,
                "ip": ip
            }
    except Exception as e:
        print(f"Error checking wifi: {e}")
        
    return {
        "connected": False,
        "ssid": None,
        "interface": "wlo1",
        "signal": "0",
        "bssid": "--",
        "security": "--",
        "ip": "--"
    }

def scan_wifi_networks() -> list:
    networks = []
    try:
        res = subprocess.run(["nmcli", "-t", "-f", "active,ssid,signal,bssid,security", "device", "wifi"], capture_output=True, text=True, timeout=8.0)
        if res.returncode == 0:
            seen_ssids = set()
            for line in res.stdout.splitlines():
                if line.startswith("Warning") or not line.strip():
                    continue
                parts = line.split(":")
                if len(parts) >= 5:
                    ssid = parts[1]
                    if not ssid:  # skip hidden
                        continue
                    active = parts[0] in ["yes", "ya"]
                    signal = parts[2]
                    bssid = parts[3].replace("\\", "")
                    security = parts[4]
                    
                    if ssid not in seen_ssids:
                        seen_ssids.add(ssid)
                        networks.append({
                            "ssid": ssid,
                            "active": active,
                            "signal": signal,
                            "bssid": bssid,
                            "security": security
                        })
    except Exception as e:
        print(f"Error scanning wifi: {e}")
    return networks

# --- WIFI ENDPOINTS ---
@app.get("/api/wifi/status")
async def api_wifi_status():
    return get_wifi_status()

@app.post("/api/wifi/scan")
async def api_wifi_scan():
    return scan_wifi_networks()

@app.post("/api/wifi/connect")
async def api_wifi_connect(request: Request):
    data = await request.json()
    ssid = data.get("ssid")
    password = data.get("password")
    if not ssid:
        raise HTTPException(status_code=400, detail="SSID is required")
    
    cmd = ["nmcli", "device", "wifi", "connect", ssid]
    if password:
        cmd += ["password", password]
        
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0)
        if res.returncode == 0:
            return {"status": "success", "message": f"Tersambung ke WiFi {ssid}"}
        else:
            return {"status": "error", "message": res.stderr.strip() or res.stdout.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

threads_started = False

def start_background_threads():
    global threads_started
    if threads_started:
        return
    threads_started = True
    print("Starting background threads...", flush=True)
    
    threading.Thread(target=alarm_monitor_loop, daemon=True).start()
    threading.Thread(target=bluetooth_auto_reconnect_loop, daemon=True).start()
    threading.Thread(target=wifi_sensing_loop, daemon=True).start()
    threading.Thread(target=camera_thread_func, daemon=True).start()

@app.on_event("startup")
async def startup_event():
    start_background_threads()

# Start script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8050, reload=False)
