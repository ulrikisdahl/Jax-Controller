import jax.numpy as jnp
import jax


class BaseController:
    def __init__(self):
        self.temp = 0.0

    
class PIDController(BaseController):
    def __init__(self, cfg: dict, key: jax.random.PRNGKey):
        self.learning_rate = cfg["learning_rate"]
        # self.key = jax.random.key(cfg["seed"])
        self.k_p = jax.random.uniform(key, shape=(1))
        self.k_i = jax.random.uniform(key, shape=(1))
        self.k_d = jax.random.uniform(key, shape=(1))
        self.error_history = 0.0
    
    @staticmethod
    def __call__(params: jax.Array, error: float, d_error: float, error_history: float):
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

    def get_params(self):
        return jnp.array([self.k_p, self.k_i, self.k_d])
    
    def get_state(self):
        return {
            "error_history": self.error_history,
            "learning_rate": self.learning_rate
        }

    def reset(self): #TODO: Could be move to a base class
        self.error_history = 0.0


class NeuralNetworkController(BaseController):
    def __init__(self, cfg: dict, key: jax.random.PRNGKey):
        self.input_dim = 3
        self.output_dim = 1 
        self.key = key
        self.learning_rate = cfg["learning_rate"]
        self.num_layers = cfg["num_layers"]
        self.num_neurons = cfg["num_neurons"]
        self.activation_fn = cfg["activation_fn"]
        self.weight_min = cfg["weight_range_low"]
        self.weight_max = cfg["weight_range_high"] 
        self.weights = None
        self._init_weights()


    def __call__(self, params: jax.Array, error: float, d_error: float, error_history: float):
        """
        x*W + b
        """
        # num_layers = params["num_layers"] #TODO: use object state variables instead
        # weights = params["weights"]
        weights = params
        activation = jnp.array([error, d_error, error_history, 1.0]) #TODO: terrible naming
        for layer in range(self.num_layers + 1): 
            layer_output = jnp.matmul(activation, weights[layer])
            if layer < self.num_layers:
                layer_output = jnp.append(layer_output, 1.0) #bias trick            
                activation = jnp.maximum(layer_output, 0) #TODO: Generalize 
        
        return layer_output #TODO: Add output function (then you wont need to clamp current_level)
    
    def _init_weights(self):
        weights = []
        in_dims = [self.input_dim] + [self.num_neurons for _ in range(self.num_layers)] 
        out_dims = [self.num_neurons for _ in range(self.num_layers)] + [self.output_dim] 
        for layer in range(self.num_layers + 1):
            weight_param = jax.random.uniform(self.key, shape=(in_dims[layer] + 1, out_dims[layer]), minval=self.weight_min, maxval=self.weight_max) # +1 for bias trick
            weights.append(weight_param)
        self.weights = weights

    def get_params(self):
        return self.weights

    def update_params(self, params: jax.Array, gradients: jax.Array):
        for layer in range(self.num_layers + 1):        
            self.weights[layer] = params[layer] - self.learning_rate * gradients[layer]