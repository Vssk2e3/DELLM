"""DELLM: Edge-Assisted DDoS Detection using a LoRA-Fine-Tuned LLM in IIoT."""

CLASSES = [
    "Benign",
    "NetDDoS",
    "AppDDoS",
    "LDDoS",
    "DrDoS",
    "Botnet",
    "MQTT_DDoS",
    "Frag_DDoS",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

__version__ = "0.1.0"
