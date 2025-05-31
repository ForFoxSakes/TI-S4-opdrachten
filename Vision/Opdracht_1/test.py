import cv2
import numpy as np

# 1. Laad de afbeeldingen
scene = cv2.imread("example6.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

# 2. Binariseer
_, scene_bin = cv2.threshold(scene, 127, 255, cv2.THRESH_BINARY)
_, template_bin = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

# 3. Vind contouren
contours_scene, _ = cv2.findContours(scene_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_template, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Check op contouren
if not contours_template:
    raise ValueError("⚠️ Geen contouren gevonden in template.png")

if not contours_scene:
    raise ValueError("⚠️ Geen contouren gevonden in scene.png")

# 5. Neem grootste contour uit template
template_contour = max(contours_template, key=cv2.contourArea)

# 6. Vergelijk met elke contour in de scene
matches = []
for cnt in contours_scene:
    score = cv2.matchShapes(template_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    matches.append((score, cnt))

# 7. Sorteer op overeenkomt en kies de beste match
matches.sort(key=lambda x: x[0])
best_score, best_match = matches[0]

print(f"Beste overeenkomst: score = {best_score:.5f}")

# 8. Teken de beste match op het beeld
scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

# Groene contour: dit is de gevonden match
cv2.drawContours(scene_color, [best_match], -1, (0, 255, 0), 2)

# Blauwe bounding box eromheen voor duidelijkheid
x, y, w, h = cv2.boundingRect(best_match)
cv2.rectangle(scene_color, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 9. Toon het resultaat
cv2.imshow("Shape match met contour en bounding box", scene_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
