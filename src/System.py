from utils import get_class, plot_mse_per_epoch
import jax.numpy as jnp
import jax
from tqdm import tqdm
from src.Plant import BasePlant
from src.Controller import BaseController



class Consys:
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

    def run(self):
        """
        """
        mse_log = []
        grad_fn = jax.value_and_grad(self.epoch_fn, argnums=0) #derivative w.r.t. controller_params
        for epoch in tqdm(range(self.epochs)):
            controller_parameters = self.controller.get_params()
            value, mse_grad = grad_fn(controller_parameters, self.key, epoch)
            self.controller.update_params(controller_parameters, mse_grad)
            print(f"Epoch {epoch}: {value}")
            mse_log.append(value)

        
        plot_mse_per_epoch(mse_log)    
        print("DONE")

    
    #@jax.jit
    def epoch_fn(
        self,
        parameters: jax.Array, #should be the only parameter, since in this function we only want to trace the parameters (to compute d_mse/d_params)  
        epoch_nr: int,
        key: jax.random.PRNGKey
    ):
        """
        Function that outputs MSE and we compute derivative w.r.t. controller parameters
        """
        self.plant.reset() #NOTE: How did this work??????
        error = 0.0
        d_error = 0.0
        error_history = 0.0 #TODO: Move to inside controller
        squared_error_history = 0.0

        noise_key = jax.lax.stop_gradient(key + epoch_nr) #NOTE 
        disturbances = jax.random.uniform(
            noise_key, shape=(self.num_timesteps),
            minval=self.noise_range_low, maxval=self.noise_range_high
        )

        error_grad_fn = jax.value_and_grad(self.step_fn, argnums=0, has_aux=True) #derivative of error w.r.t. control_signal 
        for step in range(self.num_timesteps):
            #forward 
            control_signal = self.controller(parameters, error, d_error, error_history)
            (error, output), d_error = error_grad_fn(control_signal[0], disturbances[step])
            
            #update stuff
            error_history += error
            squared_error_history += jnp.pow(error, 2)
            self.plant.update(output) 
            #TODO: controller.update(error)
            
        mse = squared_error_history / self.num_timesteps
        return mse
            
    def step_fn(
        self, 
        control_signal: float,
        disturbance: float
    ):
        """
        Outputs error and is used to compute derivative w.r.t. controller output
        """
        output, target = self.plant.evaluate(disturbance, control_signal)
        error = target - output
        return (error, output)




if __name__ == "__main__":
    pass