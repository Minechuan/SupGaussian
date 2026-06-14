"""
Visual material inference using Qwen2.5-VL-3B-Instruct.
Single-image, single-pass inference for speed.
"""

import os, sys, re, torch
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "Qwen2.5-VL-3B-Instruct")

# Single-pass prompt: visual analysis + material classification in one shot
INFER_PROMPT = """Look at this object and answer TWO questions:
1. What kind of object is this? (one sentence)
2. What material is it made of? Pick ONE from this list:
   wood | metal | plastic | rubber | ceramic | glass | cloth | leather | foam | paper | clay | stone | fruit | meat | bread | snow | sand

Reply in this exact format:
Object: <answer>
Material: <answer>"""

# Map Qwen output → material database key
MATERIAL_MAP = {
    "wood": "wood", "oak": "wood", "pine": "wood", "bamboo": "wood", "timber": "wood",
    "金属": "wood", "木头": "wood",
    "metal": "metal", "steel": "metal", "iron": "metal", "aluminum": "metal",
    "copper": "metal", "silver": "metal", "gold": "metal", "stainless": "metal",
    "plastic": "hard_plastic", "pvc": "pvc", "nylon": "nylon", "acrylic": "hard_plastic",
    "塑料": "hard_plastic",
    "rubber": "rubber", "silicone": "rubber", "elastic": "rubber",
    "橡胶": "rubber", "硅胶": "rubber",
    "ceramic": "ceramic", "porcelain": "ceramic",
    "陶瓷": "ceramic",
    "glass": "glass",
    "玻璃": "glass",
    "cloth": "cloth", "fabric": "cloth", "cotton": "cloth", "wool": "cloth", "textile": "cloth",
    "布料": "cloth", "棉": "cloth",
    "leather": "leather",
    "皮革": "leather",
    "foam": "foam", "sponge": "foam", "styrofoam": "foam",
    "泡沫": "foam", "海绵": "foam",
    "paper": "paper", "cardboard": "paper",
    "纸": "paper",
    "clay": "clay", "plasticine": "plasticine", "playdoh": "plasticine", "putty": "plasticine",
    "黏土": "plasticine", "橡皮泥": "plasticine",
    "stone": "stone", "marble": "stone", "granite": "stone", "concrete": "stone", "rock": "stone",
    "石头": "stone", "大理石": "stone",
    "fruit": "fruit", "vegetable": "fruit",
    "水果": "fruit", "蔬菜": "fruit",
    "meat": "meat", "sausage": "meat", "hotdog": "meat",
    "肉": "meat",
    "bread": "bread", "cake": "bread", "tofu": "tofu",
    "面包": "bread", "蛋糕": "bread", "豆腐": "tofu",
    "snow": "snow", "ice": "ice", "icecream": "ice_cream",
    "雪": "snow", "冰": "ice",
    "sand": "sand", "soil": "soil", "dust": "sand", "powder": "sand",
    "沙子": "sand", "土": "soil",
    "jelly": "jelly", "gelatin": "jelly", "gel": "jelly",
    "果冻": "jelly",
    "wax": "wax", "candle": "candle_wax",
    "蜡": "wax", "蜡烛": "candle_wax",
    "chocolate": "chocolate", "candy": "chocolate",
    "巧克力": "chocolate",
    "soap": "soap", "eraser": "eraser",
    "肥皂": "soap", "橡皮": "eraser",
    "lego": "hard_plastic", "toy": "hard_plastic",
}

# Object-type → material overrides (stronger signal than visual)
OBJECT_OVERRIDES = {
    "chair": "wood", "table": "wood", "desk": "wood", "shelf": "wood",
    "cabinet": "wood", "bench": "wood", "stool": "wood", "furniture": "wood",
    "drum": "metal", "microphone": "metal", "mic": "metal",
    "ship": "metal", "boat": "metal", "airplane": "metal",
    "lego": "hard_plastic", "toy": "hard_plastic",
    "hotdog": "meat", "sausage": "meat", "hamburger": "meat",
    "ficus": "wood", "plant": "wood", "tree": "wood", "flower": "wood",
    "vase": "ceramic", "cup": "ceramic", "bowl": "ceramic", "plate": "ceramic",
    "bottle": "soft_plastic", "container": "soft_plastic",
}


def load_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def infer_single_image(model, processor, image_path):
    """Single-image, single-pass material inference. Returns material_key."""
    img = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": INFER_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, temperature=0.1)

    out_ids = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()

    # Parse "Object: xxx Material: yyy"
    obj_match = re.search(r'object\s*[:：]\s*(.+?)(?:\n|$|material)', output)
    mat_match = re.search(r'material\s*[:：]\s*(.+?)$', output, re.MULTILINE)

    obj_type = obj_match.group(1).strip().rstrip('.,;') if obj_match else ""
    mat_raw = mat_match.group(1).strip().rstrip('.,;') if mat_match else output

    # Resolve material
    mat_clean = re.sub(r'[^a-z_]', '', mat_raw.replace(' ', '_').replace('-', '_'))
    material = resolve_material(mat_clean, obj_type)

    return material, obj_type, mat_raw


def resolve_material(mat_clean, obj_type=""):
    """Map model output to database key, with object-type overrides."""
    # Check object-type overrides first
    for keyword, mat in OBJECT_OVERRIDES.items():
        if keyword in obj_type:
            return mat

    # Direct lookup
    if mat_clean in MATERIAL_MAP:
        return MATERIAL_MAP[mat_clean]

    # Partial match
    for alias, mat_key in MATERIAL_MAP.items():
        if alias in mat_clean:
            return mat_key

    # Fuzzy fallbacks
    if any(w in mat_clean for w in ["metal", "steel", "iron", "aluminum", "copper", "gold", "silver"]):
        return "metal"
    if any(w in mat_clean for w in ["wood", "oak", "pine", "bamboo"]):
        return "wood"
    if any(w in mat_clean for w in ["plastic", "pvc", "nylon"]):
        return "hard_plastic"
    if any(w in mat_clean for w in ["rubber", "jelly", "gel", "silicone"]):
        return "rubber"
    if any(w in mat_clean for w in ["cloth", "fabric", "cotton", "wool"]):
        return "cloth"
    if any(w in mat_clean for w in ["glass", "ceramic", "porcelain"]):
        return "ceramic"
    if any(w in mat_clean for w in ["stone", "marble", "granite", "concrete", "rock"]):
        return "stone"
    if any(w in mat_clean for w in ["foam", "sponge"]):
        return "foam"
    if any(w in mat_clean for w in ["clay", "plasticine", "putty"]):
        return "plasticine"
    if any(w in mat_clean for w in ["leather"]):
        return "leather"
    if any(w in mat_clean for w in ["paper", "cardboard"]):
        return "paper"
    if any(w in mat_clean for w in ["snow", "ice"]):
        return "snow"
    if any(w in mat_clean for w in ["sand", "soil", "dust", "powder"]):
        return "sand"
    if any(w in mat_clean for w in ["fruit", "vegetable", "food", "meat", "bread", "cake"]):
        return "fruit"

    return "plasticine"


def infer_material_from_image(image_path, model=None, processor=None, verbose=True):
    """Single-image material inference. Returns material_key."""
    if model is None or processor is None:
        model, processor = load_model()
    material, obj_type, raw = infer_single_image(model, processor, image_path)
    if verbose:
        from material_database import get_material_params
        p = get_material_params(material)
        print(f"  Object: {obj_type} | Raw: {raw} | -> {material} (MPM: {p['material']})")
    return material
