import nest_asyncio


nest_asyncio.apply()

model_path = "/data/models/Qwen3-VL-8B-Instruct/"
chat_template = "qwen2-vl"


from io import BytesIO
import requests
from PIL import Image

from sglang.srt.parser.conversation import chat_templates

image = Image.open("example.png")


conv = chat_templates[chat_template].copy()
conv.append_message(conv.roles[0], f"What's shown here: {conv.image_token}?")
conv.append_message(conv.roles[1], "")
conv.image_data = [image]

print("Generated prompt text:")
print(conv.get_prompt())
print(f"\nImage size: {image.size}")
image

def main():
    from sglang import Engine
    llm = Engine(model_path=model_path, chat_template=chat_template, log_level="warning")
    out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
    print("Model response:")
    print(out["text"])

if __name__ == "__main__":
    main()
