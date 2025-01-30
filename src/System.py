from utils import get_class, plot_mse_per_epoch, plot_pid_parameter_history
import jax.numpy as jnp
import jax
from tqdm import tqdm
from src.Controller import PIDController
import numpy as np


class Consys:
    """
    Main class for the control system
    """
    def __init__(self, cfg: dict):
        #load config
        self.seed = cfg["seed"]
        self.num_timesteps = cfg["num_timesteps"]
        self.epochs = cfg["epochs"]
        self.noise_range_low, self.noise_range_high = cfg["noise_range_low"], cfg["noise_range_high"]
        self.key = jax.random.PRNGKey(cfg["seed"])

        #set up instances
        plant_class = get_class("src.Plant", cfg["plant"]["class"])
        self.plant = plant_class(cfg["plant"])
        controller_class = get_class("src.Controller", cfg["controller"]["class"])
        self.controller = controller_class(cfg["controller"], self.key)

        #parameter visualization
        self.k_p_history = []
        self.k_i_history = []
        self.k_d_history = [] 

    def run(self):
        """
        Main training loop
        """
        mse_log = []
        grad_fn = jax.value_and_grad(self.epoch_fn, argnums=0) #derivative w.r.t. controller_params
        for epoch in tqdm(range(self.epochs)):
            controller_parameters = self.controller.get_params()
            self.controller.reset() #NOTE: Necessary to avoid tracing leaks 
            self.plant.reset()

            value, mse_grad = grad_fn(controller_parameters, self.key, epoch)
            self.controller.update_params(controller_parameters, mse_grad)
            
            #logging
            mse_log.append(value)
            if type(self.controller) == PIDController:
                self.k_p_history.append(controller_parameters[0])
                self.k_i_history.append(controller_parameters[1])
                self.k_d_history.append(controller_parameters[2]) 
            print(f"Epoch {epoch}: {value}")
        
        plot_mse_per_epoch(mse_log)    
        plot_pid_parameter_history(self.k_p_history, self.k_i_history, self.k_d_history)

    
    def epoch_fn(
        self,
        parameters: jax.Array,
        epoch_nr: int,
        key: jax.random.PRNGKey
    ):
        """
        Function that outputs MSE and we compute derivative w.r.t. controller parameters

        Parameters
        ----------
        parameters: the controller parameters of shape
        epoch_nr: the current epoch number
        key: the random key for generating disturbances

        Returns
        -------
        mse: the mean squared error
        """
        squared_error_history = 0.0

        noise_key = jax.lax.stop_gradient(key + epoch_nr)
        disturbances = jax.random.uniform(
            noise_key, shape=(self.num_timesteps),
            minval=self.noise_range_low, maxval=self.noise_range_high
        )
        # disturbances = np.random.uniform(
        #     low=self.noise_range_low, 
        #     high=self.noise_range_high, 
        #     size=(self.num_timesteps,)
        # ).astype(np.float32)

        for step in range(self.num_timesteps):
            #forward 
            control_signal = self.controller(parameters)
            error, new_state = self.step_fn(control_signal, disturbances[step])

            #update stuff
            squared_error_history += jnp.pow(error, 2)
            self.plant.update(new_state)
            self.controller.update(error) 

        mse = squared_error_history / self.num_timesteps
        return mse
            
    def step_fn(self, control_signal: float, disturbance: float) -> tuple[float, dict]: 
        """
        Outputs error and is used to compute derivative w.r.t. controller output

        Parameters
        ----------
        control_signal: the control signal
        disturbance: the disturbance signal

        Returns
        -------
        error: the error
        new_state: the new state of the plant after the timestep
        """
        new_state = self.plant.evaluate(disturbance, control_signal)
        error = new_state["target"] - new_state["output"]
        return error, new_state




if __name__ == "__main__":
    pass




# disturbances = np.random.uniform(
#             low=self.noise_range_low, 
#             high=self.noise_range_high, 
#             size=(self.num_timesteps,)
#         ).astype(np.float32)