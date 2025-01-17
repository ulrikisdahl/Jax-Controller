import jax.numpy as jnp
import jax


class BaseController:
    def __init__(self):
        self.temp = 0.0

    
class PIDController(BaseController):
    def __init__(self, cfg: dict):
        self.learning_rate = cfg["learning_rate"]
        self.key = jax.random.key(cfg["seed"])
        self.k_p = jax.random.uniform(self.key, shape=(1))
        self.k_i = jax.random.uniform(self.key, shape=(1))
        self.k_d = jax.random.uniform(self.key, shape=(1))
        self.error_history = 0.0
    
    def __call__(self, params: jax.Array, error: float, d_error: float, error_history: float):
        """
        Computes the control signal based on the error
        Args:
            error: Error from previous timestep
            d_error: Error from previous timestep with respect to the control output
            error_history: error integral over time 
        """
        control_signal = params[0] * error + params[1] * d_error + params[2] * error_history
        return control_signal

    def update_params(self, params: jax.Array, gradients: jax.Array):
        """
        Updates the controller parameters based on the gradients
        """
        self.k_p = params[0] - self.learning_rate * gradients[0]  
        self.k_i = params[1] - self.learning_rate * gradients[1]
        self.k_d = params[2] - self.learning_rate * gradients[2]

    def update_error(self, error: float):
        self.error_history += error

    # def update_params(self, params: jax.Array):
    #     self.k_p, self.k_i, self.k_d = params

    def get_params(self):
        return jnp.array([self.k_p, self.k_i, self.k_d])
    
    def get_state(self):
        return {
            "error_history": self.error_history,
            "learning_rate": self.learning_rate
        }

    def reset(self): #TODO: Could be move to a base class
        self.error_history = 0.0