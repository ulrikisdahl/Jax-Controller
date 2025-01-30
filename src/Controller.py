import jax.numpy as jnp
import jax

class BaseController:
    """
    Base class for all controllers
    """
    def __init__(self):
        self.error = 0.0
        self.d_error = 0.0
        self.error_history = 0.0

    def update(self, new_error):
        """
        Updates the error and d_error
        """
        old_error = self.error
        self.d_error = old_error - new_error
        self.error = new_error
        self.error_history += new_error
    
    def reset(self):
        """
        Resets the controller state
        """
        self.error = 0.0
        self.d_error = 0.0
        self.error_history = 0.0

    
    
class PIDController(BaseController):
    def __init__(self, cfg: dict, key: jax.random.PRNGKey):
        super().__init__()
        self.learning_rate = cfg["learning_rate"]
        
        self.k_p = 0.01
        self.k_d = -0.01
        self.k_i = 0.0

    def __call__(self, params: jax.Array):
        """
        Computes the control signal based on the error
        
        Parameters
        ----------
        error: Error from previous timestep
        d_error: Error from previous timestep with respect to the control output
        error_history: error integral over time 
        
        Returns
        -------
        control_signal: The control signal to be applied to the plant
        """
        control_signal = params[0] * self.error + params[1] * self.d_error + params[2] * self.error_history
        return control_signal

    def update_params(self, params: jax.Array, gradients: jax.Array):
        """
        Updates the controller parameters based on the gradients
        """
        self.k_p = params[0] - self.learning_rate * gradients[0]
        self.k_i = params[1] - self.learning_rate * gradients[1]
        self.k_d = params[2] - self.learning_rate * gradients[2]

    def get_params(self):
        """
        Returns the controller parameters
        """
        return jnp.array([self.k_p, self.k_i, self.k_d])
    
    def get_state(self):
        """
        Returns the state of the controller
        """
        return {
            "error_history": self.error_history,
            "learning_rate": self.learning_rate
        }



class NeuralNetworkController(BaseController):
    def __init__(self, cfg: dict, key: jax.random.PRNGKey):
        """
        A simple feedforward MLP controller:
        """
        super().__init__()
        self.input_dim = 3 #(error, d_error, error_history)
        self.output_dim = 1 #(control signal)
        self.key = key
        self.learning_rate = cfg["learning_rate"]
        self.num_layers = cfg["num_layers"]  
        self.num_neurons = cfg["num_neurons"]
        self.activation_fn_name = cfg["activation_fn"]
        self.weight_min = cfg["weight_range_low"]
        self.weight_max = cfg["weight_range_high"]
        self.error = 0.0
        self.d_error = 0.0
        self.error_history = 0.0
        
        self._activation_map = {
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
        }
        self.activation_fn = self._activation_map.get(self.activation_fn_name, lambda x: x)
        
        #init params
        self.weights = None
        self._init_params()

    def _init_params(self) -> None:
        """
        Initialize the weights and biases of the network
        """
        dims = [self.input_dim] + [self.num_neurons for _ in range(self.num_layers - 1)] + [self.output_dim]
        
        params = []
        init_key = self.key
        
        for i in range(len(dims) - 1):
            w_key, b_key, init_key = jax.random.split(init_key, 3)
            
            weight = jax.random.uniform(w_key, shape=(dims[i], dims[i+1]),
                                   minval=self.weight_min,
                                   maxval=self.weight_max)
            bias = jax.random.uniform(b_key, shape=(dims[i+1],),
                                   minval=self.weight_min,
                                   maxval=self.weight_max)
            
            params.append((weight, bias)) 

        self.weights = params

    def __call__(self, params: list) -> jnp.ndarray:
        """
        Forward pass through the network

        Parameters
        ----------
        params: list of (weight, bias) pairs

        Returns
        -------
        logits: the output of the network
        """
        #combine inputs into a input vector
        activation = jnp.array([self.error, self.d_error, self.error_history], dtype=jnp.float32).T

        #forward pass through all layers
        for (weight, bias) in params[:-1]:
            layer_output = jnp.matmul(activation, weight) + bias
            activation = self.activation_fn(layer_output)

        #final output layer
        weight_out, bias_out = params[-1]
        logits = jnp.matmul(activation, weight_out) + bias_out
        return logits[0]

    def get_params(self):
        """
        Return the networks parameters (list of (weight, bias) pairs).
        """
        return self.weights

    def update_params(self, old_params: list, grads: list):
        """
        Update the network parameters based on the gradients
        """
        new_params = []
        for (weight, bias), (d_W, d_b) in zip(old_params, grads):
            weight_new = weight - self.learning_rate * d_W
            bias_new = bias - self.learning_rate * d_b
            new_params.append((weight_new, bias_new))
        self.weights = new_params

