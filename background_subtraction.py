import cv2 #OpenCV importieren
import numpy as np #für Morphologische Operationen (Rauschen reduzieren) - Morphologische Operationen werden in der Regel auf binäre Bilder angewendet (hier: Vordergrund weiß, Hintergrund schwarz => binär)

# Video-Datei laden
video_path = 'squirrel_vid1_cutted.mp4'
cap = cv2.VideoCapture(video_path)

# Kernel für die morphologische Operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # elliptischer Kernel der Größe 7x7

# Hintergrundsubstratkionsmodell initialisieren - Funktion createBackgroundSubstractorMOG2()
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=True) # Mixture of Gaussians (MOG): Das Modell basiert auf einer Mischung von Gauss-Verteilungen (Normalverteilungen), die auf jeden Pixel des Bildes angewendet werden. Es berücksichtigt mehrere Farben und Helligkeitswerte, um Hintergrund und Vordergrund zu modellieren. Es ist adaptiv und lernt den Hintergrund kontinuierlich. Wennn sich etwas plötzlich bewegt, dann wird es als Vordergrund erkannt.

# Prozentsatz des Bildes, das oben und unten ignoriert werden soll (40% oben und 20% unten)
crop_percent_top = 0.0
crop_percent_bottom = 0.0

# Skalierungsfaktor für das verkleinerte Bild
scale_percent = 60  # Reduziere auf 60 % der Originalgröße

while cap.isOpened():
    ret, frame = cap.read() # cap.read() gibt 2 Werte zurück: 1. ret: Bool, ob Frame erfolgreich gelesen 2. frame: tatsächlicher Frame im BGR-Format

    if not ret:
        break # wenn ret=false, also kein frame mehr vorliegt wird der Prozess beendet
    
    ############ <Bildbereich anpassen> #############
    # Bildhöhe und -breite des Frames herausfinden
    height, width = frame.shape[:2] #frame.shape gibt 3 Elemente zurück 1. Höhe, 2. Breite, 3. Kanäle -> wir extrahieren Höhe und Breite

    # Bereiche berechnen, die oben und unten abgeschnitten werden sollen
    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))

    # Frame auf den mittleren Bereich zuschneiden (ignoriere oberen und unteren Teil)
    cropped_frame = frame[top_crop:bottom_crop, :] # schneide vertikal (nach der Höhe) ab, nach dem Komma würde horizontal (nach der Breite) abgeschnitten 

    # Größe von cropped_frame verkleinern
    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))

    # Hintergrundsubstraktion anwenden
    fgmask = fgbg.apply(cropped_frame_resized) # Wir wenden die Hintergrundsubstraktion auf den Frame an 
    # Man prüft bei jedem Frame, ob der aktuelle Pixelwert zur bestehenden Hintergrundverteilung passt - wenn nicht, wird er als Vordergrund dargestellt

    fgmask[fgmask == 127] = 0  # Schatten als Hintergrund behandeln (Pixelwert 127 entspricht Schatten)

    # Rauschen mit morphologischen Operationen reduzieren
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # Öffnen: Erosion gefolgt von Dilatation: entfernt Rauschen
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # Schließen: Dilatation gefolgt von Erosion: Dilatation: schließt Lücken in Vordergrundobjekten

    # Größe des fgmask an cropped_frame anpassen (für Darstellung in einem gemeinsamen Fenster)
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)  # In 3 Kanäle konvertieren, um später beide Fenster in 1 anzeigen zu können
    fgmask_resized = cv2.resize(fgmask_color, (width_resized, height_resized))

    # Konturen finden
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #erstes Argument ist die Maske, zweites welche Konturen gefunden werden sollen (hier nur äußere), drittens welche Punkte der Konturen gespeichert werden sollten (hier komprimiert, nicht alle Punkte der Kontur)

    # Sortiere die Konturen nach ihrer Fläche und wähle die größten 5 Konturen aus
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Konturen auf dem Originalframe einzeichnen
    # -1 zeichnet alle Konturen; color und thickness passen die Linienfarbe und -dicke an
    for contour in contours:
        if 100 < cv2.contourArea(contour):  # nur Konturen mit Fläche > 100 zeichnen (damit werden sehr große Spiegelungen nicht anerkannt (allerdings geht auch Fischkontur verloren))
            cv2.drawContours(cropped_frame_resized, [contour], -1, (0, 0, 255), 2) # -1 lässt alle Konturen zeichnen 

    # Alternative ohne Mindestfläche
    # cv2.drawContours(cropped_frame_resized, contours, -1, (0, 255, 0), 2)

    # Frames nebeneinander kombinieren
    combined_frame = cv2.hconcat([cropped_frame_resized, fgmask_resized])

    # Ergebnis anzeigen
    # cv2.imshow('Original Frame', cropped_frame) # Originalvideo anzeigen
    # cv2.imshow('Foreground Mask', fgmask) # Extrahierter Vordergrund 
    cv2.imshow('Combined Frame', combined_frame)

    # Taste 'q', um Programm zu beenden
    if cv2.waitKey(30) & 0xFF == ord('q'): # Überprüfe alle 30ms, ob Taste q gedrückt wurde
        break

cap.release()
cv2.destroyAllWindows()
