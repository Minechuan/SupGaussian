"""
Extended material database for PhysGaussian MPM simulation.
MPM solver supports exactly 6 constitutive models (0-5):
  0: jelly    - elastoplastic, StVk+VM
  1: metal    - elastoplastic with hardening
  2: sand     - elastoplastic with friction angle, Drucker-Prager
  3: foam     - viscoplastic, StVk+VM, no thickening
  4: snow     - elastoplastic, critical state
  5: plasticine - elastoplastic, StVk+VM

Within each type, we vary E, nu, density, yield_stress, etc.
to create realistic presets for different real-world materials.
"""

material_database = {
    # ============ JELLY family (0): soft, highly elastic, low stiffness ============
    "jelly": {
        "material": "jelly",
        "E": 2e6, "nu": 0.40, "density": 200,
        "yield_stress": 1e4,
    },
    "gelatin": {
        "material": "jelly",
        "E": 1e6, "nu": 0.45, "density": 150,
        "yield_stress": 5e3,
    },
    "rubber": {
        "material": "jelly",
        "E": 5e6, "nu": 0.45, "density": 1100,
        "yield_stress": 2e6,
    },
    "silicone": {
        "material": "jelly",
        "E": 3e6, "nu": 0.42, "density": 1050,
        "yield_stress": 1e5,
    },
    "soft_rubber": {
        "material": "jelly",
        "E": 1e6, "nu": 0.48, "density": 900,
        "yield_stress": 5e4,
    },
    "hard_rubber": {
        "material": "jelly",
        "E": 2e7, "nu": 0.40, "density": 1200,
        "yield_stress": 5e6,
    },
    "balloon": {
        "material": "jelly",
        "E": 5e5, "nu": 0.48, "density": 100,
        "yield_stress": 2e3,
    },
    "fat": {
        "material": "jelly",
        "E": 1e5, "nu": 0.49, "density": 900,
        "yield_stress": 1e3,
    },
    "skin": {
        "material": "jelly",
        "E": 2e6, "nu": 0.48, "density": 1100,
        "yield_stress": 1e5,
    },
    "tendon": {
        "material": "jelly",
        "E": 5e7, "nu": 0.42, "density": 1200,
        "yield_stress": 5e6,
    },

    # ============ METAL family (1): stiff, high E, high density, hardening ============
    "metal": {
        "material": "metal",
        "E": 2e11, "nu": 0.30, "density": 7800,
        "yield_stress": 2e8, "hardening": 5.0,
    },
    "steel": {
        "material": "metal",
        "E": 2.1e11, "nu": 0.30, "density": 7850,
        "yield_stress": 2.5e8, "hardening": 5.0,
    },
    "aluminum": {
        "material": "metal",
        "E": 7e10, "nu": 0.33, "density": 2700,
        "yield_stress": 1.5e8, "hardening": 3.0,
    },
    "copper": {
        "material": "metal",
        "E": 1.1e11, "nu": 0.34, "density": 8960,
        "yield_stress": 7e7, "hardening": 4.0,
    },
    "iron": {
        "material": "metal",
        "E": 1.8e11, "nu": 0.29, "density": 7870,
        "yield_stress": 2e8, "hardening": 6.0,
    },
    "gold": {
        "material": "metal",
        "E": 7.9e10, "nu": 0.42, "density": 19300,
        "yield_stress": 5e7, "hardening": 2.0,
    },
    "silver": {
        "material": "metal",
        "E": 8.3e10, "nu": 0.37, "density": 10500,
        "yield_stress": 6e7, "hardening": 3.0,
    },
    "ceramic": {
        "material": "metal",
        "E": 3e11, "nu": 0.22, "density": 3800,
        "yield_stress": 5e7, "hardening": 0.0,  # brittle
    },
    "porcelain": {
        "material": "metal",
        "E": 7e10, "nu": 0.19, "density": 2400,
        "yield_stress": 3e7, "hardening": 0.0,  # brittle
    },
    "glass": {
        "material": "metal",
        "E": 7e10, "nu": 0.22, "density": 2500,
        "yield_stress": 3e7, "hardening": 0.0,  # brittle
    },
    "stone": {
        "material": "metal",
        "E": 5e10, "nu": 0.25, "density": 2700,
        "yield_stress": 1e7, "hardening": 0.5,
    },
    "marble": {
        "material": "metal",
        "E": 7e10, "nu": 0.28, "density": 2700,
        "yield_stress": 2e7, "hardening": 0.0,
    },
    "granite": {
        "material": "metal",
        "E": 5e10, "nu": 0.27, "density": 2750,
        "yield_stress": 2e7, "hardening": 0.0,
    },
    "concrete": {
        "material": "metal",
        "E": 3e10, "nu": 0.18, "density": 2400,
        "yield_stress": 3e6, "hardening": 0.0,
    },
    "brick": {
        "material": "metal",
        "E": 2e10, "nu": 0.20, "density": 2000,
        "yield_stress": 2e6, "hardening": 0.0,
    },
    "ice": {
        "material": "metal",
        "E": 9e9, "nu": 0.33, "density": 917,
        "yield_stress": 2e6, "hardening": 0.0,
    },
    "hard_plastic": {
        "material": "metal",
        "E": 3e9, "nu": 0.38, "density": 1200,
        "yield_stress": 5e7, "hardening": 2.0,
    },

    # ============ SAND family (2): granular, friction-based ============
    "sand": {
        "material": "sand",
        "E": 1e6, "nu": 0.25, "density": 1600,
        "friction_angle": 35,
    },
    "soil": {
        "material": "sand",
        "E": 5e5, "nu": 0.30, "density": 1400,
        "friction_angle": 30,
    },
    "gravel": {
        "material": "sand",
        "E": 2e6, "nu": 0.22, "density": 1800,
        "friction_angle": 42,
    },
    "dust": {
        "material": "sand",
        "E": 1e5, "nu": 0.20, "density": 800,
        "friction_angle": 20,
    },
    "powder": {
        "material": "sand",
        "E": 5e4, "nu": 0.18, "density": 600,
        "friction_angle": 15,
    },
    "salt": {
        "material": "sand",
        "E": 8e5, "nu": 0.24, "density": 1200,
        "friction_angle": 32,
    },
    "sugar": {
        "material": "sand",
        "E": 6e5, "nu": 0.23, "density": 900,
        "friction_angle": 28,
    },
    "rice": {
        "material": "sand",
        "E": 3e5, "nu": 0.22, "density": 850,
        "friction_angle": 30,
    },
    "flour": {
        "material": "sand",
        "E": 3e4, "nu": 0.15, "density": 550,
        "friction_angle": 25,
    },
    "coffee_grounds": {
        "material": "sand",
        "E": 4e5, "nu": 0.22, "density": 500,
        "friction_angle": 33,
    },

    # ============ FOAM family (3): lightweight, compressible, viscoplastic ============
    "foam": {
        "material": "foam",
        "E": 5e5, "nu": 0.10, "density": 100,
        "yield_stress": 1e4,
    },
    "sponge": {
        "material": "foam",
        "E": 1e5, "nu": 0.05, "density": 60,
        "yield_stress": 3e3,
    },
    "styrofoam": {
        "material": "foam",
        "E": 2e6, "nu": 0.12, "density": 50,
        "yield_stress": 5e4,
    },
    "cloth": {
        "material": "foam",
        "E": 1e5, "nu": 0.05, "density": 300,
        "yield_stress": 5e3,
    },
    "cotton": {
        "material": "foam",
        "E": 5e4, "nu": 0.03, "density": 80,
        "yield_stress": 2e3,
    },
    "wool": {
        "material": "foam",
        "E": 8e4, "nu": 0.05, "density": 130,
        "yield_stress": 3e3,
    },
    "paper": {
        "material": "foam",
        "E": 3e6, "nu": 0.15, "density": 800,
        "yield_stress": 2e4,
    },
    "cardboard": {
        "material": "foam",
        "E": 5e6, "nu": 0.18, "density": 700,
        "yield_stress": 5e4,
    },
    "bread": {
        "material": "foam",
        "E": 1e5, "nu": 0.08, "density": 200,
        "yield_stress": 3e3,
    },
    "cake": {
        "material": "foam",
        "E": 8e4, "nu": 0.10, "density": 400,
        "yield_stress": 2e3,
    },
    "marshmallow": {
        "material": "foam",
        "E": 2e4, "nu": 0.02, "density": 50,
        "yield_stress": 1e3,
    },
    "tofu": {
        "material": "foam",
        "E": 3e5, "nu": 0.12, "density": 700,
        "yield_stress": 5e3,
    },
    "cheese": {
        "material": "foam",
        "E": 1e6, "nu": 0.30, "density": 1100,
        "yield_stress": 1e4,
    },
    "cork": {
        "material": "foam",
        "E": 2e7, "nu": 0.10, "density": 240,
        "yield_stress": 1e5,
    },
    "leather": {
        "material": "foam",
        "E": 1e7, "nu": 0.35, "density": 860,
        "yield_stress": 5e5,
    },

    # ============ SNOW family (4): compressible, hardening when compacted ============
    "snow": {
        "material": "snow",
        "E": 1e4, "nu": 0.15, "density": 400,
        "yield_stress": 1e3, "hardening": 10.0,
    },
    "powder_snow": {
        "material": "snow",
        "E": 5e3, "nu": 0.10, "density": 100,
        "yield_stress": 5e2, "hardening": 15.0,
    },
    "wet_snow": {
        "material": "snow",
        "E": 5e4, "nu": 0.20, "density": 600,
        "yield_stress": 3e3, "hardening": 5.0,
    },
    "ice_cream": {
        "material": "snow",
        "E": 2e4, "nu": 0.25, "density": 600,
        "yield_stress": 2e3, "hardening": 8.0,
    },
    "whipped_cream": {
        "material": "snow",
        "E": 1e3, "nu": 0.05, "density": 80,
        "yield_stress": 5e2, "hardening": 20.0,
    },
    "yogurt": {
        "material": "snow",
        "E": 5e3, "nu": 0.30, "density": 1050,
        "yield_stress": 1e3, "hardening": 6.0,
    },
    "butter": {
        "material": "snow",
        "E": 5e4, "nu": 0.32, "density": 950,
        "yield_stress": 5e3, "hardening": 3.0,
    },
    "chocolate": {
        "material": "snow",
        "E": 2e6, "nu": 0.35, "density": 1300,
        "yield_stress": 2e4, "hardening": 4.0,
    },
    "wax": {
        "material": "snow",
        "E": 3e6, "nu": 0.38, "density": 900,
        "yield_stress": 1e4, "hardening": 3.0,
    },

    # ============ PLASTICINE family (5): ductile, high yield, StVk+VM ============
    "plasticine": {
        "material": "plasticine",
        "E": 5e6, "nu": 0.40, "density": 1500,
        "yield_stress": 5e5,
    },
    "clay": {
        "material": "plasticine",
        "E": 3e6, "nu": 0.42, "density": 1600,
        "yield_stress": 3e5,
    },
    "modeling_clay": {
        "material": "plasticine",
        "E": 5e6, "nu": 0.40, "density": 1500,
        "yield_stress": 5e5,
    },
    "play_doh": {
        "material": "plasticine",
        "E": 1e6, "nu": 0.45, "density": 1200,
        "yield_stress": 1e5,
    },
    "soft_plastic": {
        "material": "plasticine",
        "E": 1e8, "nu": 0.42, "density": 1000,
        "yield_stress": 2e6,
    },
    "pvc": {
        "material": "plasticine",
        "E": 3e9, "nu": 0.38, "density": 1400,
        "yield_stress": 5e7,
    },
    "nylon": {
        "material": "plasticine",
        "E": 2e9, "nu": 0.39, "density": 1140,
        "yield_stress": 7e7,
    },
    "wood": {
        "material": "plasticine",
        "E": 1e10, "nu": 0.35, "density": 700,
        "yield_stress": 5e6,
    },
    "oak": {
        "material": "plasticine",
        "E": 1.2e10, "nu": 0.32, "density": 750,
        "yield_stress": 1e7,
    },
    "pine": {
        "material": "plasticine",
        "E": 8e9, "nu": 0.30, "density": 500,
        "yield_stress": 4e6,
    },
    "bamboo": {
        "material": "plasticine",
        "E": 1.5e10, "nu": 0.30, "density": 700,
        "yield_stress": 1e7,
    },
    "wax_candle": {
        "material": "plasticine",
        "E": 1e6, "nu": 0.40, "density": 900,
        "yield_stress": 1e4,
    },
    "eraser": {
        "material": "plasticine",
        "E": 5e6, "nu": 0.42, "density": 1100,
        "yield_stress": 3e5,
    },
    "crayon": {
        "material": "plasticine",
        "E": 3e6, "nu": 0.40, "density": 900,
        "yield_stress": 2e5,
    },
    "fruit": {
        "material": "plasticine",
        "E": 5e5, "nu": 0.42, "density": 1000,
        "yield_stress": 2e4,
    },
    "vegetable": {
        "material": "plasticine",
        "E": 8e5, "nu": 0.40, "density": 950,
        "yield_stress": 3e4,
    },
    "meat": {
        "material": "plasticine",
        "E": 2e5, "nu": 0.45, "density": 1050,
        "yield_stress": 1e4,
    },
    "potato": {
        "material": "plasticine",
        "E": 1e6, "nu": 0.38, "density": 1100,
        "yield_stress": 5e4,
    },
    "banana": {
        "material": "plasticine",
        "E": 3e5, "nu": 0.43, "density": 950,
        "yield_stress": 1e4,
    },
    "apple": {
        "material": "plasticine",
        "E": 1e6, "nu": 0.40, "density": 850,
        "yield_stress": 3e4,
    },
    "soap": {
        "material": "plasticine",
        "E": 2e6, "nu": 0.38, "density": 900,
        "yield_stress": 2e5,
    },
    "candle_wax": {
        "material": "plasticine",
        "E": 8e5, "nu": 0.40, "density": 900,
        "yield_stress": 1e4,
    },
}

FALLBACK_MATERIAL = "plasticine"


def get_material_params(material_name):
    """Lookup material parameters by name (case-insensitive). Returns a copy."""
    key = material_name.lower().strip()
    if key in material_database:
        return material_database[key].copy()
    # Try common aliases
    return material_database[FALLBACK_MATERIAL].copy()


def list_materials():
    """Return all available material names."""
    return list(material_database.keys())


def list_materials_by_type(mpm_type):
    """Return materials belonging to a specific MPM type."""
    return [k for k, v in material_database.items() if v["material"] == mpm_type]


def print_all_materials():
    """Print all materials grouped by MPM type."""
    for mpm_type in ["jelly", "metal", "sand", "foam", "snow", "plasticine"]:
        mats = list_materials_by_type(mpm_type)
        print(f"\n{'='*60}")
        print(f"  {mpm_type.upper()} family ({len(mats)} variants)")
        print(f"{'='*60}")
        for name in mats:
            p = material_database[name]
            e_str = f"E={p['E']:.1e}"
            print(f"  {name:<20s} {e_str:<12s} nu={p['nu']:.2f}  rho={p['density']:.0f}", end="")
            if "yield_stress" in p:
                print(f"  yield={p['yield_stress']:.1e}", end="")
            if "friction_angle" in p:
                print(f"  friction={p['friction_angle']}°", end="")
            if "hardening" in p:
                print(f"  hardening={p['hardening']}", end="")
            print()


if __name__ == "__main__":
    print_all_materials()
    print(f"\nTotal materials: {len(material_database)}")
