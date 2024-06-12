from sim2 import *

sim = Simulation()

t = np.linspace(0,200,10001)

c1 = Gas_Tank('Throttle Chamber', 0.3, 250, 1.0)
c2 = Gas_Tank('Compress Chamber', 0.01, 250, 1.0)
c3 = Gas_Tank('Compressor Line', 0.03, 250, 1.0)
c4 = Gas_Tank('Cooled Gas', 0.03, 250, 1.0)

v1 = Gas_Valve('Throttle Valve', 1e-9, 1, 'Throttle Chamber', 'Compress Chamber')

pump2_power = np.exp(-(t-30)**2*0.005)*0.01 + 0.01

pump3_power = -np.exp(-(t-50)**2*0.01)*0.01 + 0.01
# pump3_power[10000:20000] = 0.0

v2 = Gas_Pump('Line Valve', 1e-8, 1, 'Throttle Chamber', 'Compressor Line', pump2_power)
v3 = Gas_Pump('Pump', 1e-7, 1, 'Compress Chamber', 'Compressor Line', pump3_power)

v4 = Gas_Valve('Heat Exchanger', 0, 1e3, 'Throttle Chamber', 'Cooled Gas')


sim.add_gas_tank(c1, 0.5, 0)
sim.add_gas_tank(c2, 0, 1)
sim.add_gas_tank(c3, 1, 1)
sim.add_gas_tank(c4, 0.5, -1)
sim.add_gas_valve(v1)
sim.add_gas_valve(v2)
sim.add_gas_valve(v3)
sim.add_gas_valve(v4)



sim.init_sim(t)
sim.run_sim()

semilog = False

        
plt.figure(1)
plt.clf()
fig, axs = plt.subplots(2, 3, figsize=(15,9),dpi=100, num=1)

sum_N = t*0
sum_U = t*0

axs[0, 0].set_title('Temperature (K)')
axs[1, 0].set_title('Pressure (atm)')
axs[0, 1].set_title('Number (moles)')
axs[0, 2].set_title('Flow (dVdt)')
axs[1, 1].set_title('Diagram')
axs[1, 2].set_title('Energy (J)')

for name in sim.gas_tanks:

    gas_tank_dict = sim.gas_tanks[name]
    
    x = sim.gas_tanks[name]['x']
    y = sim.gas_tanks[name]['y']
    
    sum_N += gas_tank_dict['N']
    sum_U += gas_tank_dict['U']

    axs[0, 0].plot(t,gas_tank_dict['T'],label=name)
    axs[1, 0].plot(t,gas_tank_dict['P']/101.3e3,label=name)
    axs[0, 1].plot(t,gas_tank_dict['N'],label=name)
    axs[1, 2].plot(t,gas_tank_dict['U'],label=name)
    
    axs[1, 1].scatter(x,y,gas_tank_dict['object'].V*1e4,color=(0.8,0.1,0.1))
    axs[1, 1].text(x,y,name,horizontalalignment='center')
    

for name in sim.gas_valves:
    
    gas_valve_dict = sim.gas_valves[name]
    gas_valve = sim.gas_valves[name]['object']
    
    axs[0, 2].plot(gas_valve_dict['dVdt'], label=name)
    
    x1 = sim.gas_tanks[gas_valve.tank1_name]['x']
    x2 = sim.gas_tanks[gas_valve.tank2_name]['x']
    
    y1 = sim.gas_tanks[gas_valve.tank1_name]['y']
    y2 = sim.gas_tanks[gas_valve.tank2_name]['y']
    
    axs[1, 1].plot([x1,x2],[y1,y2],color=(0.7,0.7,0.7))
    txt = axs[1, 1].text(np.mean([x1,x2]),np.mean([y1,y2]),name,horizontalalignment='center')


axs[0, 1].plot(t,sum_N,label='total')
axs[1, 2].plot(t,sum_U,label='total')
    
axs[0, 0].legend()
axs[1, 0].legend()
axs[0, 1].legend()
axs[0, 2].legend()

axs[1, 2].legend()
