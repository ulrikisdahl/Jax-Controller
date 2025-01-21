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
        """
        A simple feedforward MLP controller:
          - input_dim=3 (error, d_error, error_history)
          - output_dim=1 (control signal)
          - hidden layers determined by num_layers and num_neurons
        """
        self.input_dim = 3
        self.output_dim = 1
        self.key = key
        self.learning_rate = cfg["learning_rate"]
        self.num_layers = cfg["num_layers"]  
        self.num_neurons = cfg["num_neurons"]
        self.activation_fn_name = cfg["activation_fn"]
        self.weight_min = cfg["weight_range_low"]
        self.weight_max = cfg["weight_range_high"]
        
        self._activation_map = {
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
        }
        self.activation_fn = self._activation_map.get(self.activation_fn_name, lambda x: x)
        
        #init params
        self.weights = None
        self._init_params()

    def __call__(self, params: list, error: float, d_error: float, error_history: float) -> jnp.ndarray:
        """
        """
        #combine inputs into a input vector
        activation = jnp.array([error, d_error, error_history], dtype=jnp.float32)

        #forward pass through all layers
        for (weight, bias) in params[:-1]:
            layer_output = jnp.matmul(activation, weight) + bias
            activation = self.activation_fn(layer_output)

        #final output layer
        weight_out, bias_out = params[-1]
        logits = jnp.matmul(activation, weight_out) + bias_out
        return logits 


    def _init_params(self):
        """
        """
        dims = [self.input_dim] + [self.num_neurons for _ in range(self.num_layers)] + [self.output_dim] #possibly self.num_layers - 1

        weights = []
        for i in range(len(dims) - 1):
            weight_key, bias_key, init_key = jax.random.split(self.key, 3)

            layer_weights = jax.random.uniform(
                weight_key,
                shape=(dims[i], dims[i+1]),
                minval=self.weight_min,
                maxval=self.weight_max
            )
            bias = jax.random.uniform(
                bias_key,
                shape=(self.output_dim,),
                minval=self.weight_min,
                maxval=self.weight_max
            )

            weights.append((layer_weights, bias))
        
        self.weights = weights

    def get_params(self):
        """
        Return the networks parameters (list of (weight, bias) pairs).
        """
        return self.weights

    def update_params(self, old_params: list, grads: list):
        """
        """
        new_params = []
        for (weight, bias), (d_W, d_b) in zip(old_params, grads):
            weight_new = weight - self.learning_rate * d_W
            bias_new = bias - self.learning_rate * d_b
            new_params.append((weight_new, bias_new))
        self.weights = new_params

