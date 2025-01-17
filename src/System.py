from utils import get_class
import jax.numpy as jnp
import jax
from tqdm import tqdm
from src.Plant import BasePlant
from src.Controller import BaseController

    
def step_fn(
        control_signal: float,
        plant: BasePlant,
        plant_state: dict,
        disturbance: float
    ):
    """
    Outputs error and is used to compute derivative w.r.t. controller output
    """
    new_level, target_level = plant.evaluate(disturbance, control_signal, plant_state)
    error = target_level - new_level
    return (error[0], new_level)  #NOTE: error[0]

#@jax.jit
def epoch_fn(
        key: jax.random.PRNGKey,
        epoch_nr: int,
        noise_range: tuple[float, float], 
        num_steps: int,
        parameters: jax.Array,  
        plant: BasePlant,
        controller: BaseController
    ):
    """
    Function that outputs MSE and we compute derivative w.r.t. controller parameters
    """
    plant.reset() 
    error = 0.0
    d_error = 0.0
    error_history = 0.0 #TODO: Move to inside controller
    squared_error_history = 0.0
    disturbances = jax.random.uniform(
        key + epoch_nr, shape=(num_steps),
        minval=noise_range[0], maxval=noise_range[1]
    )

    error_grad_fn = jax.value_and_grad(step_fn, argnums=0, has_aux=True) #derivative of error w.r.t. control_signal 
    for step in range(num_steps):
        #forward 
        plant_state = plant.get_state() #NOTE: This gives state variables from Plant - might have to "grab them" before we call epoch_fn (or the whole object is a epoch_fn argument so it is pure)
        control_signal = controller(parameters, error, d_error, error_history)
        (error, new_level), d_error = error_grad_fn(control_signal, plant, plant_state, disturbances[step])
        
        #update stuff
        error_history += error
        squared_error_history += jnp.pow(error, 2)
        plant.update(new_level)
        #TODO: controller.update(error)
        
    mse = squared_error_history / num_steps
    return mse


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
        self.controller = controller_class(cfg["controller"])


    def run(self):
        """
        """
        grad_fn = jax.grad(epoch_fn, argnums=4) #derivative w.r.t. controller_params
        for epoch in tqdm(range(self.epochs)):
            controller_parameters = self.controller.get_params()
            mse_grad = grad_fn(
                self.key, epoch, (self.noise_range_low, self.noise_range_high), self.num_timesteps, controller_parameters, self.plant, self.controller
            )
            self.controller.update_params(controller_parameters, mse_grad)
            
        print("DONE")

            







if __name__ == "__main__":
    pass