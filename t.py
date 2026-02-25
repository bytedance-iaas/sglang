from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    trust_remote_code=True
)

ids = [
    6010, 311, 279, 2990, 7310, 1576, 432, 374, 5961, 8305,
    323, 7218, 13970, 13, 155678
]

print("FULL TEXT:")
print(tokenizer.decode(ids, skip_special_tokens=False))

print("\nTOKEN BY TOKEN:")
for i in ids:
    print(i, repr(tokenizer.decode([i])))
