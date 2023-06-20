from dotenv import load_dotenv
import os

load_dotenv()
folder, download_path = os.environ.get('FOLDER'), os.environ.get('DOWNLOAD')

class ImageAI:
    def save_img(self, DataBundle: object, Labeling_Dones: list) -> None:
        import time
        from urllib.request import urlretrieve
        from urllib.parse import quote
        path = f'{folder}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename[:-4]}'
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            for i, v in enumerate(Labeling_Dones):
                url = f'{download_path}/{DataBundle.bundle_uploader}/{quote(DataBundle.bundle_uploaded_filename[:-4])}/{v.data_no}'
                filepath = path + f'/{v.data_no}'
                print(url) # 디버깅 중
                if os.path.exists(filepath):
                    continue
                urlretrieve(url, filepath)
                time.sleep(0.5)
                if i == 0:
                    break
        except Exception as e:
            print(e)
            
    def convert_to_num(self, DataBundle: object) -> list:
        import glob
        files = glob.glob(f'{folder}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename[:-4]}/*.{DataBundle.bundle_data_type}')
        
        from PIL import Image
        import numpy as np
        photo_size = 32
        res = []
        for f in files:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize(size=(photo_size, photo_size)) # 32 x 32로 바꿈
            img = np.asarray(img) # 이미지를 3차원 배열로
            res.append(img) # 그걸 모아서, 학습에 사용할 것
        return res

    def color(self, array: list, Labeling_Dones: list, labels: list):
        import numpy as np
        ai_thinks = []
        answers = [[i.label] for i in Labeling_Dones]

        # 이미지, 답
        (X_train, y_train) = (np.array(array[:len(array) * 80 // 100]), np.array(answers[:len(answers) * 80 // 100]))
        (X_test, y_test) = (np.array(array[len(array) * 80 // 100:]), np.array(answers[len(answers) * 80 // 100:]))

        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        import tensorflow as tf
        from keras.utils import to_categorical
        y_train = to_categorical(y_train, len(labels))
        y_test = to_categorical(y_test, len(labels))

        from keras.layers import Dense
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

        model.fit(X_train, y_train, batch_size=120, epochs=5, validation_split=0.2)

        result = model.predict(X_test)
        
        predicts = {'예측결과': [], '실제결과': []}
        for i in range(len(result)):
            for j, acc in enumerate(result[i]):
                print(f'{labels[j]} : {round(acc * 100, 2)} %')
            print(f'예측결과 == {labels[result[i].argmax()]}')
            predicts['예측결과'].append(labels[result[i].argmax()])
            for j, res in enumerate(y_test[i]):
                if res > 0:
                    print(f'실제결과 == {labels[j]}')
                    predicts['실제결과'].append(labels[j])
        
        # 대충 이 쯤에서 정확도
        import pandas as pd
        images = [i.data_no for i in Labeling_Dones]
        labelers = [i.worked_by for i in Labeling_Dones]
        df = pd.DataFrame([(a[-12:], b, c) for a, b, c in zip(images, labelers, sum(answers, []))])
        # df.rename(columns=['image', 'labeler', 'answer'], inplace=True)
        # df.reset_index(inplace=True, drop=True)
        # df.set_index('image', inplace=True)
        # df.sort_index()
        return df
    
    def accuracy(self, correct: list, submitted: list) -> list:
        res = []
        for _ in range(len(submitted)):
            res.append(sum([1 for a, b in zip(correct, submitted) if a == b]) / len(correct))
        return res