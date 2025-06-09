import json
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

# Load model/tokenizer
model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load original data
with open("data/snips_train.json", "r") as f:
    original_data = json.load(f)["data"]

augmented_data = []
seen = set()

for item in tqdm(original_data, desc="Generating paraphrases"):
    text = item["text"].strip()
    intent = item["intent"]

    # Add original text
    if text.lower() not in seen:
        augmented_data.append({"text": text, "intent": intent})
        seen.add(text.lower())

    # Generate 10 paraphrases
    batch = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(
    **batch,
    max_length=60,
    num_return_sequences=15,   # ← Generate 15 paraphrases
    num_beams=15,              # ← Should be >= num_return_sequences
    num_beam_groups=5,         # ← Required when using diversity_penalty
    diversity_penalty=1.0,     # ← Promotes varied outputs
    do_sample=False)

    paraphrases = tokenizer.batch_decode(translated, skip_special_tokens=True)

    for para in paraphrases:
        cleaned = para.strip()
        if cleaned and cleaned.lower() not in seen:
            augmented_data.append({"text": cleaned, "intent": intent})
            seen.add(cleaned.lower())

# Save augmented dataset
with open("data/snips_augmented.json", "w") as f:
    json.dump({"data": augmented_data}, f, indent=2)

print(f"✅ Dataset expanded to {len(augmented_data)} examples.")
