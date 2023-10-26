import cv2
import face_recognition

                                          # images should be placed in folder named recognized_faces


def recognize_faces_from_camera(known_faces, known_names):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)

        # Encode the faces in the current frame
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Iterate through each face encoding in the frame
        for face_encoding in face_encodings:
            # Compare the face encoding with the known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            name = "Unknown"

            # Check if there is a match
            if True in matches:
                # Find the index of the matched face
                matched_index = matches.index(True)

                name = known_names[matched_index]

            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Recognized Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Load known face images for recognition


known_face_images = [face_recognition.load_image_file("./recognized_faces/face1.jpg"),
            face_recognition.load_image_file("./recognized_faces/face2.jpg")]  # Add more images here if required

known_face_encodings = [face_recognition.face_encodings(image)[0] for image in known_face_images]
print(known_face_encodings)

# Create a list of known faces and corresponding names
known_faces = known_face_encodings
known_names = ["Name of person1", "Name of person2"]  # Add more names here if required in the same order as images

# Call the recognize_faces_from_camera function
recognize_faces_from_camera(known_faces, known_names)
