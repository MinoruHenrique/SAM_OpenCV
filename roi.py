import cv2

# Cargar imagen
image = cv2.imread("imagen.jpg")

# Mostrar imagen
cv2.imshow("Imagen", image)

rects = []
while True:
    # Seleccionar región con un rectángulo
    rect = cv2.selectROI("Imagen", image, False)

    # Mostrar región seleccionada con un rectángulo
    x, y, w, h = rect
    rects.append(rect)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Imagen", image)
    if cv2.waitKey(0) & 0xFF == ord('q'): #save on pressing 'y' 
            cv2.destroyAllWindows()
            print(rects)
            break
