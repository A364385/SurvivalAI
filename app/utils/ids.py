import uuid

def generate_id(prefix: str = "") -> str:
    """Generates a unique ID, optionally with a prefix."""
    base_id = str(uuid.uuid4())
    return f"{prefix}_{base_id}" if prefix else base_id
