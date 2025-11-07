import json
import os

SESSION_FILE = "data/sessions.json"


def load_sessions():
    """Load sessions from disk or start fresh."""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: sessions.json corrupted, starting fresh.")
    return {}


def save_sessions(session_store):
    """Save all sessions to disk."""
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(session_store, f, indent=2)


def list_sessions(session_store):
    """Return a list of all active session IDs."""
    return list(session_store.keys())


def get_session(session_store, session_id):
    """Return messages for a given session."""
    return session_store.get(session_id, [])


def delete_session(session_store, session_id):
    """Remove a specific session."""
    if session_id in session_store:
        del session_store[session_id]
        save_sessions(session_store)
        return True
    return False
