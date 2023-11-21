import sys
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

valid = False

if len(sys.argv) > 2:
    car_spd = sys.argv[1]
    meters_diff = sys.argv[2]
    if 30 <= car_spd <= 150 and 0 <= meters_diff <= 300:
        valid = True
    else:
        print("[ERROR] ERROR IN PARAMETERS \nfirst parameter(car_spd) should be between 30 and 150 \nsecond parameter should be between 0 and 300")
if not valid:
    car_spd = 30
    meters_diff = 250

print("[INFO] USING DEFAULT VALUES")

print("[INFO] Car speed:{}/150".format(car_spd))

print("[INFO] meters of difference:{}/300".format(meters_diff))

x_car_spd = np.arange(30, 150, 5)

x_meters_diff = np.arange(0, 300, 5)

x_braking = np.arange(0, 100, 5)

func_car_spd_very_slow = fuzz.trimf(x_car_spd, [30, 30, 60])

func_car_spd_slow = fuzz.trimf(x_car_spd, [30, 60, 90])

func_car_spd_normal = fuzz.trimf(x_car_spd, [60, 90, 120])

func_car_spd_fast = fuzz.trimf(x_car_spd, [90, 120, 150])

func_car_spd_very_fast = fuzz.trimf(x_car_spd, [120, 150, 150])

func_meters_few = fuzz.trimf(x_meters_diff, [0, 0, 150])

func_meters_normal = fuzz.trimf(x_meters_diff, [0, 150, 300])

func_meters_many = fuzz.trimf(x_meters_diff, [150, 300, 300])

func_braking_null = fuzz.trimf(x_braking, [0, 0, 50])

func_braking_light = fuzz.trimf(x_braking, [0, 50, 100])

func_braking_strong = fuzz.trimf(x_braking, [50, 100, 100])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_car_spd, func_car_spd_very_slow, 'brown', linewidth=2, label='Very Slow')

ax0.plot(x_car_spd, func_car_spd_slow, 'b', linewidth=2, label='Slow')

ax0.plot(x_car_spd, func_car_spd_normal, 'green', linewidth=2, label='Normal')

ax0.plot(x_car_spd, func_car_spd_fast, 'yellow', linewidth=2, label='Fast')

ax0.plot(x_car_spd, func_car_spd_very_fast, 'red', linewidth=2, label='Very Fast')

ax0.set_title('Car Speed')

ax0.legend()

ax1.plot(x_meters_diff, func_meters_few, 'b', linewidth=2, label='Short Distance')

ax1.plot(x_meters_diff, func_meters_normal, 'g', linewidth=2, label='Acceptable Distance')

ax1.plot(x_meters_diff, func_meters_many, 'r', linewidth=2, label='Long Distance')

ax1.set_title('Distance to the car in front')

ax1.legend()

ax2.plot(x_braking, func_braking_null, 'r', linewidth=1.5, label='braking null')

ax2.plot(x_braking, func_braking_light, 'g', linewidth=1.5, label='braking light')

ax2.plot(x_braking, func_braking_strong, 'b', linewidth=1.5, label='braking strong')

ax2.set_title('Braking Coeficient')

ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()

spd_very_slow = fuzz.interp_membership(x_car_spd, func_car_spd_very_slow, car_spd)

spd_slow = fuzz.interp_membership(x_car_spd, func_car_spd_slow, car_spd)

spd_normal = fuzz.interp_membership(x_car_spd, func_car_spd_normal, car_spd)

spd_fast = fuzz.interp_membership(x_car_spd, func_car_spd_fast, car_spd)

spd_very_fast = fuzz.interp_membership(x_car_spd, func_car_spd_very_fast, car_spd)

meters_few = fuzz.interp_membership(x_meters_diff, func_meters_few, meters_diff)

meters_normal = fuzz.interp_membership(x_meters_diff, func_meters_normal, meters_diff)

meters_many = fuzz.interp_membership(x_meters_diff, func_meters_many, meters_diff)

rule1_without_clipping = np.fmax(spd_very_slow, meters_many)

rule1 = np.fmin(rule1_without_clipping, func_braking_null)

rule2_without_clipping = np.fmax(

np.fmin(np.fmax(meters_normal, meters_many), spd_slow),

np.fmin(np.fmax(spd_slow, np.fmax(spd_normal, np.fmax(spd_fast, spd_very_fast))), meters_normal))

rule2 = np.fmin(rule2_without_clipping, func_braking_light)

rule3_without_clipping = np.fmin(meters_few, np.fmax(spd_normal, np.fmax(spd_fast, spd_very_fast)))

rule3 = np.fmin(rule3_without_clipping, func_braking_strong)

#Solo para visualización

ceros = np.zeros_like(x_braking)

fig, ax0 = plt.subplots(figsize=(10, 3))

ax0.fill_between(x_braking, ceros, rule1, facecolor='b', alpha=0.7)

ax0.plot(x_braking, func_braking_null, 'b', linewidth=0.5, linestyle='-', )

ax0.fill_between(x_braking, ceros, rule2, facecolor='g', alpha=0.7)

ax0.plot(x_braking, func_braking_light, 'g', linewidth=0.5, linestyle='-')

ax0.fill_between(x_braking, ceros, rule3, facecolor='r', alpha=0.7)

ax0.plot(x_braking, func_braking_strong,'r', linewidth=0.5, linestyle='-')

ax0.set_title('Rules')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()

aggregated = np.fmax(rule1,

np.fmax(rule2, rule3))

braking = fuzz.defuzz(x_braking, aggregated, 'centroid')

print("[SOLUTION] {}".format(braking))

# Para visualización

activacion = fuzz.interp_membership(x_braking, aggregated, braking)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_braking, func_braking_null, 'b', linewidth=0.5, linestyle='-', )

ax0.plot(x_braking, func_braking_light, 'g', linewidth=0.5, linestyle='-')

ax0.plot(x_braking, func_braking_strong, 'r', linewidth=0.5, linestyle='-')

ax0.fill_between(x_braking, ceros, aggregated, facecolor='Orange', alpha=0.7)

ax0.plot([braking, braking], [0, activacion], 'k', linewidth=1.5, alpha=0.9)

ax0.set_title('Result')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()

plt.show()