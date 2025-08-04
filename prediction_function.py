import numpy as np
import pickle

class InferenceTimePredictor:
    """
    A class for predicting inference times using a pre-trained model.
    It encapsulates methods for predicting individual layer inference times
    and total inference times for a series of layers.
    """
    def __init__(self, model_path: str):
        """
        Initialises the InferenceTimePredictor by loading the pre-trained model.

        Args:
            model_path: The file path to the pickled model.
        """
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Please ensure it exists.")
            self.model = None # Set model to None if loading fails
        except Exception as e:
            print(f"An error occurred while loading the model from {model_path}: {e}")
            self.model = None

    def predict(self, i: int, o: int) -> float:
        """
        Predicts the inference time given x and y using the loaded model.

        Args:
            x: Input value x (e.g., input size of a layer).
            y: Input value y (e.g., output size of a layer).

        Returns:
            The predicted inference time.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Cannot perform prediction.")
        
        # The model's predict method expects a NumPy array, even if it ignores the input
        prediction = self.model.predict(np.array([[i, o]])) 
        return prediction

    # def predict_inference_time_nl(self, layers: list, c: float = 1.95) -> float:
    #     """
    #     Predicts the total inference time given a list of layer sizes using the loaded model.
    #     It sums the predicted inference times for each sequential pair of layers,
    #     adding a constant 'c' for each layer transition.

    #     Args:
    #         layers: A list of layer sizes.
    #                 For example: [4, 64, 64, 64, 64, 64, 64, 1] would imply:
    #                 layer 1 with 4 inputs, 64 outputs;
    #                 subsequent layers have 64 inputs and outputs;
    #                 the final layer has 64 inputs and 1 output.
    #         c: A constant value to be added per layer transition. Best value for now is 1.95.

    #     Returns:
    #         The predicted total inference time.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model is not loaded. Cannot perform prediction.")

    #     total_inf_time = sum(
    #         self.predict(layers[i], layers[i + 1]) + c
    #         for i in range(len(layers) - 1)
    #     )
    #     return total_inf_time

# --- ReLULatencyModel (as provided, stays separate as it's the model definition) ---
class ReLULatencyModel:
    """
    A simple model that always returns an expected latency of 2 for a ReLU layer.
    """
    def __init__(self, c):
        self.model = c

    def predict(self, data):
        """
        This method simulates a prediction, always returning 2.
        The 'data' parameter is included for consistency with typical model interfaces,
        but it is not used in this specific model.
        """
        return self.model

# --- Model Creation and Saving (unchanged) ---
# relu_model = ReLULatencyModel(c=2)
# fc_model = InferenceTimePredictor(model_path="pretrained_regression_models/Polynomial Regression.pkl")

# # # Define the filename for your PKL file
# relu_pkl_filename = "STM32F746NG/relu.pkl"
# fc_pkl_filename = "STM32F746NG/fc.pkl"

# # Save the model to a PKL file
# try:
#     with open(relu_pkl_filename, 'wb') as file:
#         pickle.dump(relu_model, file)
#     print(f"\nModel successfully saved to {relu_pkl_filename}")

#     with open(fc_pkl_filename, 'wb') as file:
#         pickle.dump(fc_model, file)
#     print(f"Model successfully saved to {fc_pkl_filename}")
# except Exception as e:
#     print(f"An error occurred while saving the model: {e}")

# --- Usage Example ---
if __name__ == "__main__":
    relu_pkl_filename = "STM32F746NG/relu.pkl"
    fc_pkl_filename = "STM32F746NG/fc.pkl"
    polynomial = "pretrained_regression_models/Polynomial Regression.pkl"
    random_forest = "pretrained_regression_models/Random Forest Regression.pkl"
    
    relu_predictor = ReLULatencyModel(c=1.95)
    fc_predictor = InferenceTimePredictor(model_path=polynomial)

    # Example usage of predict_inference_time
    if relu_predictor.model and fc_predictor.model: # Check if model loaded successfully
        single_layer_latency = fc_predictor.predict(128, 256)
        relu_latency = relu_predictor.predict(data=[128, 256])
        print(f"\nPredicted inference time for a single layer (128x256): {single_layer_latency + relu_latency}")

        # Example usage of predict_total_inference_time_nl
        # example_layers = [4, 64, 64, 64, 64, 64, 64, 1]
        # constant_c = 1.95
        # total_latency = fc_predictor.predict_inference_time_nl(example_layers, constant_c)
        # print(f"Predicted total inference time for layers {example_layers} with c={constant_c}: {total_latency}")
