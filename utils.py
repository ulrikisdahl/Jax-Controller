import importlib
import matplotlib.pyplot as plt

def get_class(module_name: str, class_name: str):
    """
    Dynamically imports and retreives classes
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except:
        raise ImportError(f"Could not import module {module_name}")

def plot_mse_per_epoch(mse_array):
    plt.figure(figsize=(8, 5))
    plt.plot(mse_array, marker='o', linestyle='-', color='b', label='MSE')
    plt.title("Mean Squared Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_pid_parameter_history(k_p_history, k_d_history, k_i_history):
    """
    Plots the history of PID controller parameters over epochs.
    """
    epochs = range(len(k_p_history))

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, k_p_history, label="k_p", marker='o')
    plt.plot(epochs, k_i_history, label="k_d", marker='s')
    plt.plot(epochs, k_d_history, label="k_i", marker='^')
    
    plt.xlabel("Epochs")
    plt.ylabel("Parameter Value")
    plt.title("PID Parameter History")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()