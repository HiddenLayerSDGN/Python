class ImageAI:
    def save_img(self, DataBundle: object, Labeling_Dones: list) -> None:
            import os, time
            if not os.path.exists(f'{원하는 경로}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename}'):
                os.makedirs(f'{원하는 경로}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename}')

            from urllib.request import urlretrieve
            try:
                for i, v in enumerate(Labeling_Dones):
                    url = f'{다운받을 경로}/{DataBundle.bundle_uploader}/{DataBundle.bundle_folder_name}/{v.data_no}'
                    print(DataBundle.bundle_folder_name) # 이 부분 문제있는 거 같음
                    filepath = f"{원하는 경로}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename}/{i + 1}.jpg"
                    if os.path.exists(filepath):
                        continue
                    urlretrieve(url, filepath)
                    time.sleep(0.5)
            except Exception as e:
                print(e)

    def convert_to_num(self, DataBundle: object) -> list:
        import glob
        files = glob.glob(f'{원하는 경로}/{DataBundle.bundle_uploader}/{DataBundle.bundle_uploaded_filename}/*.jpg')

        from PIL import Image
        import numpy as np
        photo_size = 32
        res = []
        for f in files:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize(photo_size, photo_size) # 32 x 32로 바꿈
            img = np.asarray(img) # 이미지를 3차원 배열로
            res.append(img) # 그걸 모아서, 학습에 사용할 것
        return res

    def black_white(self):
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # 이 부분 바꾸고
        
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # 흑백이라
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        from keras.models import Sequential
        from keras import layers
        model = Sequential()
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax')) # x개

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        
        model.fit(X_train, y_train, epochs=10, validation_split=0.2)
        
        model.evaluate(X_test, y_test)
        
        results = model.predict(X_test)

        import numpy as np
        from sklearn.metrics import classification_report
        print(classification_report(np.argmax(y_test, axis = -1), np.argmax(results, axis = -1)))

    def color(self, array: list, Labeling_Dones: list):
        import numpy as np
        answers = [[i.label] for i in Labeling_Dones]
        (X_train, y_train) = (np.array(array[:len(array) * 80 / 100]), np.array(answers[:len(answers) * 80 / 100]))
        (X_test, y_test) = (np.array(array[:len(array) * 20 / 100]), np.array(answers[:len(answers) * 20 / 100]))
        # 이미지, 답

        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        from tensorflow.keras.utils import to_categorical
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        import tensorflow as tf
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
        Dense(10, activation='softmax')])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        model.fit(X_train, y_train, batch_size=120, epochs=5, validation_split=0.2)

        result = model.predict(X_test[0].reshape(-1, 32, 32, 3))[0]

        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i, acc in enumerate(result):
            print(labels[i], ':', round(acc * 100, 2), '%')
        print('예측결과', labels[result.argmax()])

        for i, res in enumerate(y_test[0]):
            if res > 0:
                print('실제결과', labels[i])

        # 대충 이 쯤에서 정확도

        return {'예측결과': labels[result.argmax()], '실제결과': labels[i]}
    
    def accuracy(self, correct: list, submitted: list):
        res = []
        for _ in range(len(submitted)):
            res.append(sum([1 for a, b in zip(correct, submitted) if a == b]) / len(correct))
        return res