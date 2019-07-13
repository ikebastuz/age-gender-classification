from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

class AgeGenderClassifier():
    def __init__(self, img_size, weights_file):
        self.img_size = img_size
        self.weights_file = weights_file
        
    def __call__(self):
        model = MobileNet(weights='imagenet', include_top=False,
                        input_shape=(self.img_size, self.img_size, 3))
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.15)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.15)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.15)(x)
        age_preds = Dense(7, activation='softmax', name="pred_age")(x)
        gender_preds = Dense(2, activation='softmax', name="pred_gender")(x)
        model = Model(inputs=model.input, outputs=[gender_preds, age_preds])
        model.load_weights(self.weights_file)

        return model
    