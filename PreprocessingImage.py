from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from face_recognition import face_locations, load_image_file, face_landmarks
import cv2
import numpy as np
from imutils import paths
import os
from sklearn.preprocessing import LabelEncoder

'''class tiền xử lý ảnh để chuẩn bị dự liệu cho việc train'''
class PreprocessingFaceImg:
    def __init__(self):
        self.SIZE = (64,128)
        self.SOCHIEU = 3780
        self.pixels_per_cell = (8,8)
        self.cells_per_block = (2,2)
    '''xử lý tất cả các ảnh trong thư mực chứa ảnh train hoặc test'''
    def ManyFaceFeatures(self,derectory):
        faces = list()
        path = list(paths.list_images(derectory))
        print("Dataset: "+str(len(path))+" images")
        ''' tạo một vector 0 để sau đó gán các đặc trừng khuôn mặt vào vector x này. làm như này sẽ
            ta có được 1 vector đặc trưng phù hợp với yêu cầu đầu vào của model svm mà k cần reshape '''
        x = np.zeros((self.SOCHIEU, len(path))) #tạo một mảng 2 chiều với giá trị 0
        count = 0
        try:
            for i in range(len(path)):
                img = cv2.imread(path[i])
                box = face_locations(img) # tìm tất cả các khuôn mặt có trong ảnh
                if len(box) == 0:
                    return None
                else:
                    top, right, bottom, left = box[0]
                    face = img[top:bottom, left:right]
                    face_resize = cv2.resize(face, self.SIZE)
                    # Use HOG to extract features of face images
                    fd, hog_image = hog(face_resize, orientations=9, pixels_per_cell= self.pixels_per_cell,cells_per_block= self.cells_per_block,visualize=True,transform_sqrt=True, multichannel=True)
                    x[:, i] = fd
                    count +=1
                    print(str(count) + " images was trained ==> "+ path[i])
        except:
            print("Anh so " + count+ " loi!"+ path[i])

        labels = [p.split(os.path.sep)[-2] for p in path]
        return np.asarray(x), np.asarray(labels)
        '''trả về một mảng các đặc trưng của từng bức khuôn mặt train và
        nhãn tương ứng của từng khuôn mặt train'''

    '''trích xuất đặc trừng của ảnh test. hàm này giống với hàm trích xuất đặc trưng ở class đầu nhưng khác là
        class đầu làm việc với thư mục còn hàm này làm việc trực tiếp với ảnh test'''
    def FaceFeatures(self,img_test):
        box = face_locations(img_test)
        if len(box) == 0:  # nếu k detect được khuôn mặt hoặc k có khuôn mặt trong màn hình thì return ảnh gốc
            return img_test, box
        else:
            #phần này để lấy đặc trưng của tất cả các khuôn mặt trong ảnh
            face_arr = []
            for i in box:
                x1, y1, x2, y2 = i
                img_crop = img_test[x1:x2, y2:y1]
                img_resize = cv2.resize(img_crop, self.SIZE)
                img_array = np.asarray(img_resize)
                fd, hog_image = hog(img_resize, orientations=9, pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block, visualize=True, multichannel=True)
                # print(fd.shape)
                face_arr.append(fd.T) # chuyển ma trận từ code về hàng
            face_arr = np.asarray(face_arr)
            return face_arr, box

    def encoder_label(self,labels):
        ''' hàm mã hóa labels thành các số nguyên (0,1,2...) để quá trình train và xử lý của máy tính được nhanh
         hơn là các string (bill, midu..)'''
        label_encoder = LabelEncoder()
        label = label_encoder.fit_transform(labels)
        return label    # trả về một mảng lables có các phần tử là các số nguyên ([0,0,0,0,1,1,1...])

'''build và dự đoán kết quả của ảnh test'''
class ModelAndPrediction:
    def __init__(self):
        self.C = 100
        self.gamma = 1
        self.kernel = 'poly'
        self.degree = 3
    def BuiltModel(self, XTrain,YTrain):  # xây dựng thuật toán SVM
        model = SVC(kernel=self.kernel, degree= self.degree, gamma=self.gamma, C=self.C)
        model.fit(XTrain, YTrain)
        return model

class DataTest(PreprocessingFaceImg,ModelAndPrediction):
    def test():
        pass

if __name__ == '__main__':
    # M = ModelAndPrediction()
    # pre = PreprocessingFaceImg()
    # img_test = cv2.imread("bill_er2.jpg")
    # face,box = pre.FaceFeatures(img_test)
    # MAndP = ModelAndPrediction()
    # pre = PreprocessingFaceImg()
    # label_encoder = LabelEncoder()
    # data = np.load('img.npz')
    # x_train, x_label = data['arr_0'], data['arr_1']
    # data_test = np.load('2_persons_test.npz')
    # y_test, y_label = data_test['arr_0'], data['arr_1']
    # print(x_train)
    # x_label = label_encoder.fit_transform(x_label)
    # # img_test = cv2.imread("bill_er2.jpg")
    # model = MAndP.BuiltModel(x_train, x_label)
    # feature, boxs = pre.FaceFeatures(img_test)
    # result = model.predict(feature)
    # result = np.asarray(result)
    # name = label_encoder.inverse_transform(result)

    # y_pred = model.predict(y_test)
    # print(y_pred)


    '''tiền xử lý dữ liệu rồi lưu thành 1 file npz để thuận tiện cho việc train và update model'''
    pre = PreprocessingFaceImg()
    X_train, labels = pre.ManyFaceFeatures('E:/FaceRecognition/image_train')
    X_train = X_train.T
    np.savez_compressed('img.npz', X_train, labels)

