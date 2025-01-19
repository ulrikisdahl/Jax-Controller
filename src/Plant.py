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

    def evaluate(self, disturbances: float, control_signal: float):
        """
        Computes the new water level in the bathtub given the disturbances and control signal
        Returns:
            self.current_level: the new water level in the bathtub after the timestep
            self.initial_level: the initial water level in the bathtub
        """
        self.current_level = jnp.maximum(self.current_level, 1e-6)

        velocity = jnp.sqrt(2 * self.gravitational_constant * self.current_level)
        flow_rate =  jnp.multiply(velocity, self.cross_sec_C)  

        volume_change = control_signal + disturbances - flow_rate
        delta_height = jnp.divide(volume_change, self.cross_sec_A) #the change in water height

        new_level = self.current_level + delta_height
        return (new_level, self.target_level)
        

    def update(self, new_level: float):
        self.current_level = new_level #self.plant is never traced in the System, so we can mutate its state variables all we want!

    def get_state(self) -> tuple:
        return {
            "current_level": self.current_level,
            "target_level": self.target_level, 
            "cross_sec_A": self.cross_sec_A, 
            "cross_sec_C": self.cross_sec_C
        }
    
    def reset(self):
        self.current_level = self.target_level


class CournotPlant(BasePlant):
    def __init__(self, cfg: dict):
        super(CournotPlant).__init__()
        self.amount_q1 = 0.0
        self.amount_q2 = 0.0
        self.p_max = cfg["p_max"]
        self.marginal_cost = cfg["marginal_cost"] #c_m
        self.profit_goal = cfg["profit_goal"] #T

    def evaluate(self, disturbances: float, control_signal: float):
        """
        """
        #q(t+1) = U + q(t)
        amount_q1 = control_signal + self.amount_q
        
        #q(t+1) = D + q(t)
        amount_q2 = disturbances + self.amount_q2

        #q = q1 + q1
        total_amount = amount_q1 + amount_q2

        #p(q) = p_max - q
        price = self.p_max - total_amount

        #P1 = q * (p(q) - c_m)
        profit = amount_q1 * (price - self.marginal_cost)

        return profit, self.profit_goal #E = T - P1
    

    def update(self):
        self.amount_q2

    def reset(self):
        self.amount_q1 = 0.0
        self.amount_q2 = 0.0

if __name__ == "__main__":          
    obj = BathtubPlant()
    obj()


