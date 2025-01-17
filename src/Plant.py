import jax
import jax.numpy as jnp

class BasePlant:
    def __init__(self):
        self.temp = 0.0


class BathtubPlant(BasePlant):
    def __init__(self, cfg: dict):
        super(BathtubPlant).__init__()
        self.target_level = cfg["initial_level"] #starting level of bathtub
        self.current_level = 0.0 #current level which we aim to keep close to initial_level
        self.timestep_length = 1 #sec 
        self.gravitational_constant = jnp.divide(9.8, jnp.pow(self.timestep_length, 2)) 
        self.cross_sec_A = cfg["cross_sec_A"]
        self.cross_sec_C = cfg["cross_sec_C"] #cross-sectional area of drain

    def evaluate(self, disturbances: float, control_signal: float, plant_state: dict):
        """
        Computes the new water level in the bathtub given the disturbances and control signal
        Returns:
            self.current_level: the new water level in the bathtub after the timestep
            self.initial_level: the initial water level in the bathtub
        """
        current_level, target_level, cross_sec_A, cross_sec_C = plant_state["current_level"], plant_state["target_level"], plant_state["cross_sec_A"], plant_state["cross_sec_C"]
        timestamp_length = 1 #sec
        gravitational_constant = jnp.divide(9.8, jnp.pow(timestamp_length, 2)) 

        velocity = jnp.sqrt(2 * gravitational_constant * current_level)
        flow_rate =  jnp.multiply(velocity, cross_sec_C)  

        volume_change = control_signal + disturbances - flow_rate
        delta_height = jnp.divide(volume_change, cross_sec_A) #the change in water height

        new_level = current_level + delta_height 
        return (new_level, target_level)
    
    def update(self, new_level: float):
        self.current_level = new_level

    def get_state(self) -> tuple:  
        return {
            "current_level": self.current_level,
            "target_level": self.target_level, 
            "cross_sec_A": self.cross_sec_A, 
            "cross_sec_C": self.cross_sec_C
        }
    
    def reset(self):
        self.current_level = self.target_level


if __name__ == "__main__":          
    obj = BathtubPlant()
    obj()


