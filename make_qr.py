import qrcode

# ✏️ Replace this with your actual Hugging Face Space link
# For example: "https://billymarietta-cityproject.hf.space"
url = "https://YOUR-HUGGINGFACE-LINK.hf.space"

# Generate the QR code
img = qrcode.make(url)

# Save it to your project folder
img.save("huggingface_app_qr.png")

print("✅ QR code created successfully! Saved as huggingface_app_qr.png")
