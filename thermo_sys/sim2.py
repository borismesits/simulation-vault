import numpy as np
import matplotlib.pyplot as plt

k = 8.3  

f = 3

a = 12
b = 0.001

def idealgas_P(T, V, N):

    return N*k*T/V


def idealgas_U(N, T, V):
    
    f = 3
    
    return T*N*k*f/2

def idealgas_T(N, U, V):
    
    f = 3
    
    if N>0:
        return 2*U/(f*N*k)
    else: 
        return 0

def vanderwaals_U(N, T, V):
    
    f = 3
    
    return f/2*N*k*T- a*N**2/V

def vanderwaals_T(N, U, V):
    
    f = 3
    
    if N>0:
        return 2*(U + a*N**2/V)/(f*N*k)
    else:
        return 0

def vanderwaals_P(T, V, N):
    
    return N*k*T/(V-N*b) - a*N**2/V**2


f_T2P = vanderwaals_P
f_T2U = vanderwaals_U
f_U2T = vanderwaals_T

# f_T2P = idealgas_P
# f_T2U = idealgas_U
# f_U2T = idealgas_T


class Simulation():
    
    def __init__(self):
        
        self.gas_tanks = {}
        self.heat_reserviors = {}
        self.gas_valves = {}

        
    def add_gas_tank(self, gas_tank, x, y):
        
        name = gas_tank.name
        
        dictt = {'object': gas_tank, 'x': x, 'y': y}
        
        self.gas_tanks[name] = dictt
        
    def init_sim(self, t):
        
        self.t = t
        
        for name in self.gas_tanks:
            
            gas_tank_dict = self.gas_tanks[name]
            gas_tank = gas_tank_dict['object']
            
            gas_tank_dict['U'] = t*0
            U0 = f_T2U(gas_tank.N0, gas_tank.T0, gas_tank.V)
            
            gas_tank.U = U0
            
            gas_tank_dict['N'] = t*0
            gas_tank.N = gas_tank.N0

            gas_tank_dict['T'] = t*0
            gas_tank.T = gas_tank.T0
            
            P0 = f_T2P(gas_tank.T0, gas_tank.V, gas_tank.N0)
            
            gas_tank_dict['P'] = t*0
            gas_tank.P = P0
        
            
        for name in self.gas_valves:
            
            gas_valve = self.gas_valves[name]['object']
            
            self.gas_valves[name]['dVdt'] = t*0
            gas_valve.dVdt = 0
            
            gas_valve.tank1_dict = self.gas_tanks[gas_valve.tank1_name]
            gas_valve.tank2_dict = self.gas_tanks[gas_valve.tank2_name]
            
            gas_valve.tank1 = self.gas_tanks[gas_valve.tank1_name]['object']
            gas_valve.tank2 = self.gas_tanks[gas_valve.tank2_name]['object']
            
    def add_heat_reservoir(self, heat_reservior):
        
        pass
    
    def add_gas_valve(self, gas_valve):
        
        name = gas_valve.name
        
        dictt = {'object': gas_valve}
        
        self.gas_valves[name] = dictt
        
    def run_sim(self):
        
        dt = self.t[1] - self.t[0]
        
        for i in range(0, len(self.t)):
            
        
            for name in self.gas_valves:
                
                gas_valve = self.gas_valves[name]['object']
                
                gas_valve.passive_flow(i, dt)
                
                gas_valve.tank1_dict['U'][i] = gas_valve.tank1.U
                
                gas_valve.tank1_dict['U'][i] = gas_valve.tank1.U
                gas_valve.tank1_dict['N'][i] = gas_valve.tank1.N
                gas_valve.tank1_dict['T'][i] = gas_valve.tank1.T
                gas_valve.tank1_dict['P'][i] = gas_valve.tank1.P
                
                gas_valve.tank2_dict['U'][i] = gas_valve.tank2.U
                gas_valve.tank2_dict['N'][i] = gas_valve.tank2.N
                gas_valve.tank2_dict['T'][i] = gas_valve.tank2.T
                gas_valve.tank2_dict['P'][i] = gas_valve.tank2.P
                
                self.gas_valves[name]['dVdt'][i] = gas_valve.dVdt

class Gas_Tank():
    def __init__(self, name, V, T0=300, N0=0):
        
        self.name = name
        self.T0 = T0
        self.N0 = N0
        self.V = V
        
class Heat_Reservior():
    def __init__(self, C, T):
        
        self.C = C
        self.T = T
        
class Gas_Valve():
    def __init__(self, name, K_N, K_T, tank1_name, tank2_name):
        
        self.name = name
        
        self.K_N = K_N
        self.K_T = K_T
        
        self.tank1_name = tank1_name
        self.tank2_name = tank2_name
        
    def f_dVdt(self, P1,P2,i):
        
        return (P1-P2)*self.K_N
        
        
    def passive_flow(self, i, dt):
        
        N1 = self.tank1.N
        N2 = self.tank2.N
        
        V1 = self.tank1.V
        V2 = self.tank2.V
        
        U1 = self.tank1.U
        U2 = self.tank2.U

        T1 = f_U2T(N1, U1, V1)
        T2 = f_U2T(N2, U2, V2)
        
        P1 = f_T2P(T1, V1, N1)
        P2 = f_T2P(T2, V2, N2)
        
        if P1 < 0:
            P1 = 0
        if P2 < 0:
            P2 = 0
        
        dVdt = self.f_dVdt(P1,P2,i)
        
        self.dVdt = dVdt
        
        dN1dt = 0
        dN2dt = 0
        
        heat_cond = -(T1 - T2)*self.K_T*dt # just temperature change from conduction
        
        if dVdt > 0:
            
            dN1dt = -dVdt/V1 * N1
            dN2dt = dVdt/V1 * N1
            
            dU1dt = -dVdt/V1*U1 + heat_cond
            dU2dt = dVdt/V1*U1 - heat_cond


        elif dVdt < 0:
            
            dN1dt = -dVdt/V2 * N2
            dN2dt = dVdt/V2 * N2
            
            dU1dt = -dVdt/V2*U2 + heat_cond
            dU2dt = dVdt/V2*U2 - heat_cond
            
        else:
            
            dU1dt = heat_cond
            dU2dt = -heat_cond


        self.tank1.U = U1 + dU1dt*dt
        self.tank2.U = U2 + dU2dt*dt
        
        self.tank1.N = N1 + dN1dt*dt
        self.tank2.N = N2 + dN2dt*dt
        
        self.tank1.T = T1
        self.tank2.T = T2
        
        self.tank1.P = P1
        self.tank2.P = P2
        
class Heat_Exchanger():
    def __init__(self, K_T, tank, heat_reservior):
        
        self.K_T = K_T
        self.tank = tank
        self.heat_reservior = heat_reservior
        
class Gas_Pump(Gas_Valve):
    def __init__(self, name, K_N, K_T, tank1, tank2, dVdt_p):
        super().__init__(name, K_N, K_T, tank1, tank2)
        
        self.dVdt_p = dVdt_p
        
    def f_dVdt(self, P1,P2,i):
        
        return (P1-P2)*self.K_N + self.dVdt_p[i]
        
class Turbo_Gas_Pump(Gas_Pump):
    def __init__(self, K_N, K_T, tank1, tank2, power):
        super().__init__(K_N, K_T, tank1, tank2)
        
        self.power = power
        self.speed = 0
        
    def update(self, dt):
        
        pass
        




