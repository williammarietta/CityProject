import qrcode, sys
url = sys.argv[1] if len(sys.argv)>1 else input("HTTPS URL: ").strip()
img = qrcode.make(url)
img.save("qr_app_link.png")
print("Saved: qr_app_link.png")
