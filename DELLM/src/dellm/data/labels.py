"""Map each dataset's raw attack labels to DELLM's 8 classes.

Rules are applied in order; the first match wins. Labels that match a DROP rule
(non-DDoS attacks such as single-source DoS, recon, brute force, web attacks)
return None and are removed. Check the printed mapping report for every dataset:
dataset label spellings vary between versions.
"""
import re

from .. import CLASSES

# (regex, class or None). Lower-case matching, applied in order.
RULES = [
    # --- benign ---
    (r"^(benign|normal|benigntraffic|benign_final)$", "Benign"),
    (r"benign", "Benign"),
    # --- non-DDoS attacks we drop (single-source DoS, recon, etc.) ---
    (r"^dos[-_ ]", None),
    (r"recon|scan|sweep|discovery|brute|dictionary|password|sql|xss|injection|"
     r"upload|hijack|backdoor|ransom|mitm|arp|spoof|fingerprint|vulnerab|exploit", None),
    # --- DDoS families (order matters) ---
    (r"frag", "Frag_DDoS"),
    (r"mirai|botnet|bot_|^bot$|ares|byob|zeus", "Botnet"),
    (r"mqtt", "MQTT_DDoS"),
    (r"slow|loris|lddos|low[-_ ]?rate|shrew", "LDDoS"),
    (r"^drdos|reflect|amplif", "DrDoS"),
    (r"dns|ntp|ldap|mssql|netbios|snmp|ssdp|tftp|portmap|chargen|memcache|cldap", "DrDoS"),
    (r"http|web|application|cc[-_ ]attack|get[-_ ]flood|post[-_ ]flood", "AppDDoS"),
    (r"syn|udp|icmp|tcp|ack|rst|fin|psh|synonymous|lag|flood|ddos", "NetDDoS"),
]


def map_label(raw, extra_rules=None):
    """Return one of CLASSES, or None if the label should be dropped."""
    s = str(raw).strip().lower()
    # benign rules first, then dataset-specific rules, then the general rules
    for pattern, cls in RULES[:2] + [tuple(r) for r in (extra_rules or [])] + RULES[2:]:
        if re.search(pattern, s):
            return cls
    return None


def mapping_report(raw_labels, extra_rules=None):
    """Return {raw_label: mapped_class} for a quick manual check."""
    return {lab: map_label(lab, extra_rules) for lab in sorted(set(map(str, raw_labels)))}


assert set(c for _, c in RULES if c) <= set(CLASSES)
