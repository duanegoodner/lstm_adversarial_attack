from datetime import datetime

def generate_session_id() -> str:
    return "".join(char for char in str(datetime.now()) if char.isdigit())
