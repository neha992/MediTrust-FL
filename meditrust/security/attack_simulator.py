import numpy as np
def attack(update, attack_type, strength):
    strength=float(strength)
    if attack_type == "sign_reversal": return -np.asarray(update)*strength
    if attack_type == "scaled_update": return np.asarray(update)*strength
    raise ValueError("Unknown attack")
