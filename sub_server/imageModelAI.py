from oracleDB import oracleDB
from dotenv import load_dotenv
import os, time, glob
from urllib.request import urlretrieve
from urllib.parse import quote
from PIL import Image
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense
from collections import Counter, defaultdict

load_dotenv()
db = oracleDB(os.environ.get('ID'), os.environ.get('PW'), os.environ.get('IP'), os.environ.get('PORT'), os.environ.get('SID'))
folder, download_path = os.environ.get('FOLDER'), os.environ.get('DOWNLOAD')

def calc_accuracy(df: list, images: list, answers: list) -> list:
    worker_score = defaultdict(int)
    checklist = {img_no: correct_answer for img_no, correct_answer in zip(images, answers)}
    for _, v in df.iterrows():
        if checklist[v.image] == int(v.answer):
            worker_score[v.labeler] += 1
    return [(float(worker_score[worked_by] / len(df) * 100), worked_by) for worked_by in worker_score.keys()]

class ImageAI:
    def save_img(self, project_no: int, DataBundle: object, Labeling_Dones: list) -> list:
        path = f'{folder}/{project_no}'
        if not os.path.exists(path):
            os.makedirs(path)
        array = []
        try:
            for v in Labeling_Dones:
                url = f'{download_path}/{DataBundle.bundle_uploader}/{quote(DataBundle.bundle_uploaded_filename[:-4])}/{v.data_no}'
                filepath = path + f'/{v.data_no}'
                array.append([v.data_no[-12:-4], v.worked_by, v.label]) # image 번호, 작업자, 고른 답
                if os.path.exists(filepath):
                    continue
                urlretrieve(url, filepath)
        except Exception as e:
            print(e)
        return array
                
    def convert_to_num(self, project_no: int, DataBundle: object) -> list:
        files = glob.glob(f'{folder}/{project_no}/*.{DataBundle.bundle_data_type}')
        photo_size = 32
        res = []
        for f in files:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize(size=(photo_size, photo_size)) # 32 x 32로 바꿈
            img = np.asarray(img) # 이미지를 3차원 배열로
            res.append(img) # 그걸 모아서, 학습에 사용할 것
        return res

    def color(self, project_no: int, array: list, convert_images_3D: list, labels: list):
        print('작업 시작')
        df = pd.DataFrame(array)
        df.rename(columns={0: 'image', 1: 'labeler', 2: 'answer'}, inplace=True)
        
        if len(convert_images_3D) < len(array):
            images = list(map(lambda x: x[0], array[:len(convert_images_3D)])) # 사진 갯수만큼 가져온 것
            answers = [Counter(list(df[df['image'] == str(i)]['answer'].astype('int32'))).most_common(1)[0][0] for i in images] # 통계로 voting에 가깝게 구현함
            # 답이, 사진 갯수보다 적을 때는, 라벨링이 덜 된 것이니 그냥 원본그대로 가면 되고
            # 답이, 사진 갯수보다 많을 때는 사진 갯수만큼 가져와서 == 마지막 -> 처음 되기 직전 부분까지 가져와서 통계 
        else:
            images = list(df['image'])
            answers = list(df['answer'].astype('int32'))

        db.insert_result(project_no, images, answers) # 정산하기 누르면 의뢰자에게 건네줄 데이터 db에 저장
        
        # 이미지, 답
        X_train, X_test, y_train, y_test = train_test_split(convert_images_3D, answers, test_size=0.2, random_state=0) # 원본의 타입이 유지됨

        X_train = np.array(X_train).astype('float32') / 255
        X_test = np.array(X_test).astype('float32') / 255

        y_train = to_categorical(y_train, len(labels))
        y_test = to_categorical(y_test, len(labels))

        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(len(labels), activation='softmax')])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        model.fit(X_train, y_train, batch_size=120, epochs=10, validation_split=0.2)

        result = model.predict(X_test)
        
        predicts = {'예측결과': [], '실제결과': []}
        for i in range(len(result)):
            for j, acc in enumerate(result[i]):
                print(f'{labels[j]} : {round(acc * 100, 2)} %')
            print(f'예측결과 : {labels[result[i].argmax()]}')
            predicts['예측결과'].append(labels[result[i].argmax()])
            for j, res in enumerate(y_test[i]):
                if res > 0:
                    print(f'실제결과 : {labels[j]}')
                    predicts['실제결과'].append(labels[j])

        db.update_trust(calc_accuracy(df, images, answers)) # [(신뢰도, 작업자), ...]를 넣어줌 

        return predicts