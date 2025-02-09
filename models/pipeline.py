import cv2
import shap
import joblib
import requests
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timezone

from lime.lime_tabular import LimeTabularExplainer
from tensorflow import keras
from tensorflow.keras.activations import softmax

API_KEY = "4112acf218e91354a5c5722563befae6"

class Model:
    def __init__(self, vgg_model, rf_model, X_train_data):
        self.model = keras.models.load_model(vgg_model)
        self.rf_model = joblib.load(rf_model)
        self.shap_explainer = shap.TreeExplainer(self.rf_model)
        X_train = np.load(X_train_data)
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=['prcp', 'prcp_avg_3d', 'blockage'],
            class_names=['No Flood', 'Flood'],
            categorical_features=[2],
            mode='classification'
        )
        self.data = None

    def __clean_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        return image.reshape(1, 256, 256, 3)

    def __get_past_weather(self, lat, lon):
        prcp_2d = []
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - 86400
        url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_KEY}"
        data = requests.get(url).json()
        prcp_2d += [np.sum([d.get('rain', {'1h': 0})['1h'] for d in data.get('list', [])])]

        end = start
        start -= 86400
        data = requests.get(url).json()
        prcp_2d += [np.sum([d.get('rain', {'1h': 0})['1h'] for d in data.get('list', [])])]

        return prcp_2d

    def __get_weather(self, lat, lon):
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
        data = requests.get(url).json()

        prcp = data.get('rain', {'1h': 0})['1h']

        prcp_2d = self.__get_past_weather(lat, lon)
        prcp_avg_3d = np.mean(prcp_2d + [prcp])

        return prcp, prcp_2d, prcp_avg_3d

    def explain(self):
        shap_values = self.shap_explainer.shap_values(self.data)
        shap_values = shap_values[:,0]
        shap_values = np.array(shap_values).reshape(3, 1)

        prcp, prcp_2d, prcp_avg_3d = self.__get_weather(20, 20)
        plot_list = []

        # Precipitation History Plot
        fig1, ax1 = plt.subplots()
        ax1.plot(["Day -2", "Day -1", "Today"], prcp_2d[::-1] + [prcp], marker='o')
        ax1.set_title("Precipitation History")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Rainfall (mm)")
        ax1.set_ylim(0, None)
        ax1.grid()
        plot_list.append(fig1)

        # Precipitation Overview Plot
        fig2, ax2 = plt.subplots()
        ax2.bar(["Current Prcp", "3-Day Avg Prcp"], [prcp, prcp_avg_3d])
        ax2.set_title("Precipitation Overview")
        ax2.set_ylabel("Rainfall (mm)")
        plot_list.append(fig2)

        # LIME Explanation plot
        explanation = self.lime_explainer.explain_instance(self.data, self.rf_model.predict_proba)
        fig3 = explanation.as_pyplot_figure()
        plot_list.append(fig3)

        # SHAP Force Plot
        plt.figure()
        shap.force_plot(self.shap_explainer.expected_value[1], shap_values[:, 0], feature_names=['Prcp', 'Prcp_3d_avg', 'Blockage'], matplotlib=True)
        plt.tight_layout()
        plot_list.append(plt)

        return plot_list

    def predict(self, image, lat, lon):
        image = self.__clean_image(image)
        pred = self.model.predict(image)
        blockage = np.argmax(softmax(pred), axis=1)[0]
        prcp, _, prcp_avg_3d = self.__get_weather(lat, lon)
        self.data = np.array([prcp, prcp_avg_3d, blockage])
        pred = self.rf_model.predict([self.data])
        plot_list = self.explain()
        return pred[0], plot_list


if __name__ == '__main__':
    m = Model('vgg16_model.keras', 'rf_model.joblib', 'X_train_smote.npy')

    image = cv2.imread('1.jpg')
    p, i = m.predict(image, 20, 20) # value either 0 or 1
    print(p)
    # 0 - No Flood
    # 1 - Flood

    # For CNN Model
    # 0 = No Blockage
    # 1 = Partial Blockage
    # 2 = Full Blockage
