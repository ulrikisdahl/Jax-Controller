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
        
        Parameters
        ----------
        disturbances: the disturbances in the system
        control_signal: the control signal from current timestep

        Returns
        -------
        self.current_level: the new water level in the bathtub after the timestep
        self.initial_level: the initial water level in the bathtub
        """
        self.current_level = jnp.maximum(self.current_level, 1e-6)

        velocity = jnp.sqrt(2 * self.gravitational_constant * self.current_level)
        flow_rate =  jnp.multiply(velocity, self.cross_sec_C)  

        volume_change = control_signal + disturbances - flow_rate
        delta_height = jnp.divide(volume_change, self.cross_sec_A) #the change in water height

        new_level = self.current_level + delta_height
        return {
            "output": new_level,
            "target": self.target_level
        }        

    def update(self, new_state: dict):
        self.current_level = new_state["output"] #self.plant is never traced in the System

    def reset(self):
        self.current_level = self.target_level


class CournotPlant(BasePlant):
    def __init__(self, cfg: dict):
        super(CournotPlant).__init__()
        self.amount_q1 = 0.5
        self.amount_q2 = 0.5
        self.p_max = cfg["p_max"]
        self.marginal_cost = cfg["marginal_cost"] #c_m
        self.profit_goal = cfg["profit_goal"] #T

    def evaluate(self, disturbances: float, control_signal: float) -> dict:
        """

        Parameters
        ----------
        disturbances: the disturbances in the system
        control_signal: the control signal from current timestep

        Returns
        -------
        profit: the profit of the firm
        profit_goal: the profit goal of the firm
        """
        #q(t+1) = U + q(t)
        amount_q1 = control_signal + self.amount_q1
        amount_q1 = jnp.clip(amount_q1, min=0.0, max=1.0)
        # amount_q1 = control_signal + jnp.clip(self.amount_q1, min=0.0, max=1.0)

        #q(t+1) = D + q(t)
        amount_q2 = disturbances + self.amount_q2
        amount_q2 = jnp.clip(amount_q2, min=0.0, max=1.0)
        # amount_q2 = control_signal + jnp.clip(self.amount_q2, min=0.0, max=1.0)

        #q = q1 + q1
        total_amount = amount_q1 + amount_q2

        #p(q) = p_max - q
        price = jnp.maximum(self.p_max - total_amount, 0.0)

        #P1 = q * (p(q) - c_m)
        profit = amount_q1 * (price - self.marginal_cost)

        # return profit, self.profit_goal #E = T - P1
        return {
            "output": profit,
            "target": self.profit_goal,
            "amount_q1": amount_q1,  
            "amount_q2": amount_q2
        } 
    
    def update(self, new_state: dict):
        """
        Updates the state of the plant
        """
        self.amount_q1 = new_state["amount_q1"]
        self.amount_q2 = new_state["amount_q2"]

    def reset(self):
        """
        Resets the state of the plant
        """
        self.amount_q1 = 0.5
        self.amount_q2 = 0.5



class ProductionPlant(BasePlant): #resource allocation
    def __init__(self, cfg: dict):
        super(ProductionPlant).__init__()
        self.target = cfg["target_volume"]
        self.Q = 0.5 #initial production
        self.k = cfg["production_efficiency"]
        self.c = cfg["decay_rate"]


    def evaluate(self, disturbances: float, control_signal: float):
        """
        dQ/dt = kR - cQ
        
        Parameters
        ----------
        Q: Production output
        control_signal: resources allocated (R)
        k: Production efficiency
        c: Decay rate (machinery wear)

        Returns
        -------
        Q_new: the new production output
        target: the target production
        """
        dQ_dt = self.k * control_signal - self.c * self.Q
        Q_new = self.Q + dQ_dt + disturbances
        return {
            "output": Q_new,
            "target": self.target
        }
    
    def update(self, new_state: dict):
        """
        Updates the state of the plant
        """
        self.Q = new_state["output"]

    def reset(self):
        """
        Resets the state of the plant
        """
        self.Q = 0.5




class CarVelocityPlant(BasePlant): #cruise control
    def __init__(self, cfg: dict):
        super(CarVelocityPlant).__init__()
        self.drag = cfg["drag"]
        self.friction = cfg["friction"]
        self.target = cfg["target_velocity"]
        self.initial_velocity = cfg["initial_velocity"]
        self.velocity = self.initial_velocity
        self.dt = 1.0

    def evaluate(self, disturbances: float, control_signal: float):
        """
        """
        engine_force = control_signal
        
        drag_force = self.drag * (self.velocity**2)

        net_force = engine_force - drag_force - self.friction + disturbances

        new_velocity = self.velocity + self.dt * net_force
        new_velocity = jnp.maximum(new_velocity, 0.0)

        return {
            "output": new_velocity,
            "target": self.target
        }
    
    def update(self, new_state: dict):
        self.velocity = new_state["output"]

    def reset(self):
        self.velocity = self.initial_velocity


if __name__ == "__main__":          
    obj = BathtubPlant()
    obj()


