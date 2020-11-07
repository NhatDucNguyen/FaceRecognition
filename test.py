from PreprocessingImage import *

MAndP = ModelAndPrediction()
pre = PreprocessingFaceImg()
label_encoder = LabelEncoder()
data = np.load('img.npz')
x_train, x_label = data['arr_0'], data['arr_1']
x_label = label_encoder.fit_transform(x_label)
model = MAndP.BuiltModel(x_train, x_label)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    feature, boxs = pre.FaceFeatures(frame)
    try:
        result = model.predict(feature)
        # name = label_encoder.inverse_transform(result)
        if len(boxs) == 0:
            cv2.imshow('View face', frame)
        else:
            for box,feature in zip(boxs, feature): # chay dong thoi 2 vong lap 1 luc
                feature = feature.reshape(-1, 3780) # dua ve  dang vector
                result = model.predict(feature) # du doan dac trung truyen vao so voi anh train
                name = label_encoder.inverse_transform(result)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (y2, x1), (y1, x2), (255,255,255), 1)
                cv2.putText(frame, name[0],(y2, x1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1) #viet text
                cv2.imshow('View face', frame)
    except:
        cv2.imshow('View face',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # giai phong du lieu video
cv2.waitKey()
cv2.destroyAllWindows() # giai phong du lieu anh
