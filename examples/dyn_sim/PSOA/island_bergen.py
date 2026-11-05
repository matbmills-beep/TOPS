import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import importlib
import warnings
importlib.reload(dps)
import numpy as np


def step_and_detect_hygov_overflow(solver, overflow_time):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        solver.step()

    for warning in caught:
        warnings.showwarning(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
            file=warning.file,
            line=warning.line,
        )
        if (
            overflow_time[0] is None
            and warning.category is RuntimeWarning
            and warning.filename.endswith('/gov.py')
            and 'overflow encountered' in str(warning.message)
        ):
            overflow_time[0] = solver.t


def add_hygov_overflow_marker(axes, overflow_time):
    if overflow_time[0] is None:
        print('No HYGOV overflow warning detected.')
        return

    print('HYGOV overflow first detected at t = {:.3f} s.'.format(overflow_time[0]))
    for axis in axes.flat:
        axis.axvline(
            overflow_time[0],
            color='red',
            linestyle='--',
            #label='HYGOV overflow',
        )


def apply_ufls(ps, measured_frequency, UFLS_flag):
    if not hasattr(ps, '_ufls_state'):
        UFLS_stages = [False] * 10
        UFLS_thresholds = [49.0 - 0.15 * index for index in range(10)]
        ps._ufls_state = {
            'enabled': UFLS_flag,
            'stages': UFLS_stages,
            'thresholds': UFLS_thresholds,
            'admittance_step': 0.0175,
        }

    ufls_state = ps._ufls_state
    if not ufls_state['enabled']:
        return

    for index, threshold in enumerate(ufls_state['thresholds']):
        if not ufls_state['stages'][index] and measured_frequency < threshold:
            ufls_state['stages'][index] = True

    active_stages = [
        index + 1 for index, tripped in enumerate(ufls_state['stages']) if tripped
    ]
    if active_stages:
        bus_admittance_change = -ufls_state['admittance_step'] * max(active_stages)
        ps.y_bus_red_mod[32, 32] = bus_admittance_change
        ps.y_bus_red_mod[38, 38] = bus_admittance_change


def ufls_shed_power_mw(ps, t, x):
    ufls_state = ps._ufls_state
    stage_count = max(
        (index + 1 for index, tripped in enumerate(ufls_state['stages']) if tripped),
        default=0,
    )
    if stage_count == 0:
        return 0.0

    voltage = ps.solve_algebraic(t, x)
    bus_indices = (32, 38)
    voltage_squared_sum = sum(abs(voltage[index]) ** 2 for index in bus_indices)
    return (
        ufls_state['admittance_step']
        * stage_count
        * voltage_squared_sum
        * ps.s_n
    )


def transformer_from_current(ps, x, v, transformer_name='T5510-5560'):
    transformer = ps.trafos['Trafo']
    transformer_index = int(np.flatnonzero(
        transformer.par['name'] == transformer_name)[0])
    return abs(transformer.I_from(x, v)[transformer_index])


def general(t_end=15):
    UFLS_flag = True

    # Load model
    import tops.ps_models.n45_2025 as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    overflow_time = [None]
    res = defaultdict(list)
    t_0 = time.time()

    bergen_gen = np.array([31, 32, 36])
    bergen_gen_names = ps.gen['GEN'].par['name'][bergen_gen]
    bergen_bus_mask = np.isin(
        ps.buses['name'],
        ['5510', '5560']
    )

    bergen_load = np.array([31, 32])
    bergen_load_names = ps.loads['Load'].par['name'][bergen_load]

    print('Active power imbalance:', np.sum(ps.gen['GEN'].par['P'][bergen_gen]), '-', np.sum(ps.loads['Load'].par['P'][bergen_load]),
        '=', np.sum(ps.gen['GEN'].par['P'][bergen_gen]) - np.sum(ps.loads['Load'].par['P'][bergen_load]))

    island_flag = True
    #UFLS filter settings
    T_m = 0.15 # Time constant for measurement in seconds
    f_meas = ps.f_n # Nominal frequency in Hz

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Bergen is islanded
        if t > 1 and island_flag:
            island_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][50], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][7], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][14], 'disconnect')

        # Simulate next step
        step_and_detect_hygov_overflow(sol, overflow_time)
        x = sol.y
        v = sol.v
        t = sol.t

        # Under frequence load shedding
        f_actual = 50 * (1 + (ps.gen['GEN'].speed(x, v)[32] + ps.gen['GEN'].speed(x, v)[31] + ps.gen['GEN'].speed(x, v)[36])/3)
        dt = 5e-3
        f_meas = f_meas + (dt/T_m) * (f_actual - f_meas)

        apply_ufls(ps, f_meas, UFLS_flag)

        # Store result
        res['t'].append(t)
        gen_speed = ps.gen['GEN'].speed(x, v)
        gen_angle = ps.gen['GEN'].angle(x, v)
        frequency_hz = 50 * (1 + gen_speed[bergen_gen])
        relative_angle = np.rad2deg(gen_angle[bergen_gen] - gen_angle[36])
        mechanical_power = ps.gen['GEN'].P_m(x, v)[bergen_gen] * ps.gen['GEN'].par['S_n'][bergen_gen]
        electrical_power = ps.gen['GEN'].P_e(x, v)[bergen_gen]
        power_imbalance = mechanical_power - electrical_power
        bergen_voltage = np.abs(v[bergen_bus_mask])

        res['frequency_hz'].append(frequency_hz.copy())
        res['relative_angle_deg'].append(relative_angle.copy())
        res['mechanical_power_mw'].append(np.sum(mechanical_power))
        res['electrical_power_mw'].append(np.sum(electrical_power))
        res['power_imbalance_mw'].append(np.sum(power_imbalance))
        res['voltage_min_pu'].append(np.min(bergen_voltage))
        res['voltage_max_pu'].append(np.max(bergen_voltage))
        res['transformer_current_ka'].append(transformer_from_current(ps, x, v))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    time_values = np.asarray(res['t'])
    frequency_values = np.asarray(res['frequency_hz'])
    frequency_nadir = np.min(frequency_values)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

    axes[0, 0].plot(time_values, frequency_values)
    axes[0, 0].axhline(frequency_nadir, color='black', linestyle='--',
                        label=f'Nadir: {frequency_nadir:.2f} Hz')
    axes[0, 0].set_ylabel('Frequency [Hz]')
    axes[0, 0].set_title('Bergen generator frequency')
    axes[0, 0].legend(list(bergen_gen_names) + [f'Nadir: {frequency_nadir:.2f} Hz'])

    axes[0, 1].plot(time_values, np.asarray(res['relative_angle_deg']))
    axes[0, 1].set_ylabel('Relative rotor angle [deg]')
    axes[0, 1].set_title('Rotor angles relative to G5560-1')
    axes[0, 1].legend(bergen_gen_names)

    axes[1, 0].plot(time_values, res['mechanical_power_mw'], linestyle='--')
    axes[1, 0].plot(time_values, res['electrical_power_mw'])
    axes[1, 0].plot(time_values, res['power_imbalance_mw'], color='black')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Power [MW]')
    axes[1, 0].set_title('Bergen generator power response')
    axes[1, 0].legend(['Mechanical', 'Electrical', 'P imbalance'])

    axes[1, 1].plot(time_values, res['voltage_min_pu'], label='Minimum')
    axes[1, 1].plot(time_values, res['voltage_max_pu'], label='Maximum')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Voltage [p.u.]')
    axes[1, 1].set_title('Bergen bus-voltage envelope')
    axes[1, 1].legend()

    axes[2, 0].plot(time_values, res['transformer_current_ka'])
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('Current [kA]')
    axes[2, 0].set_title('T5510-5560 from-side current (5510 side)')
    axes[2, 0].legend(['T5510-5560'])
    axes[2, 1].set_visible(False)

    #add_hygov_overflow_marker(axes, overflow_time)
    axes[0, 0].legend(list(bergen_gen_names) + [f'Nadir: {frequency_nadir:.2f} Hz'])
    axes[0, 1].legend(list(bergen_gen_names))
    axes[1, 0].legend(['Mechanical', 'Electrical', 'P imbalance'])
    axes[1, 1].legend(['Minimum', 'Maximum'])

    for axis in axes.flat:
        axis.grid()

    fig.suptitle('Bergen response after islanding')
    fig.tight_layout()
    plt.show()



def load_change(lc, case_label, col, t_end=15, UFLS_flag=False, plot_all=True):

    # Load model
    import tops.ps_models.n45_2025 as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Changing the load
    load_table = model['loads']
    p_index = load_table[0].index('P')
    load_row_1 = next(row for row in load_table[1:] if row[0] == lc[0][0])
    load_row_1[p_index] = lc[0][1]  # MW
    load_row_2 = next(row for row in load_table[1:] if row[0] == lc[1][0])
    load_row_2[p_index] = lc[1][1]  # MW

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    overflow_time = [None]
    res = defaultdict(list)
    t_0 = time.time()

    generator_names = ps.gen['GEN'].par['name']
    bergen_gen_names = ['G5510-1', 'G5510-2', 'G5560-1']
    bergen_gen = np.array([
        int(np.flatnonzero(generator_names == name)[0])
        for name in bergen_gen_names
    ])
    frequency_gen_idx = int(np.flatnonzero(generator_names == 'G5560-1')[0])
    angle_gen_idx = int(np.flatnonzero(generator_names == 'G5510-1')[0])
    angle_reference_idx = frequency_gen_idx
    voltage_bus_idx = int(np.flatnonzero(ps.buses['name'] == '5560')[0])

    bergen_load = np.array([31, 32])
    bergen_load_names = ps.loads['Load'].par['name'][bergen_load]
    ufls_bus_indices = np.array([32, 38])
    ufls_bus_names = ps.buses['name'][ps.bus_idx_red[ufls_bus_indices]]
    print('UFLS buses (reduced indices 32, 38):', ufls_bus_names)

    print('Active power imbalance:', np.sum(ps.gen['GEN'].par['P'][bergen_gen]), '-', np.sum(ps.loads['Load'].par['P'][bergen_load]),
        '=', np.sum(ps.gen['GEN'].par['P'][bergen_gen]) - np.sum(ps.loads['Load'].par['P'][bergen_load]))

    island_flag = True
    #UFLS filter settings
    T_m = 0.6 # Time constant for measurement in seconds
    f_meas = ps.f_n # Nominal frequency in Hz

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Bergen is islanded
        if t > 1 and island_flag:
            island_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][50], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][7], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][14], 'disconnect')

        # Simulate next step
        step_and_detect_hygov_overflow(sol, overflow_time)
        x = sol.y
        v = sol.v
        t = sol.t

        # Under frequence load shedding
        f_actual = 50 * (1 + ps.gen['GEN'].speed(x, v)[frequency_gen_idx])
        dt = 5e-3
        f_meas = f_meas + (dt/T_m) * (f_actual - f_meas)

        apply_ufls(ps, f_meas, UFLS_flag)
        res['ufls_shed_mw'].append(ufls_shed_power_mw(ps, t, x))


        #if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            #print(f"\nUgyldige verdier (NaN/inf) oppstod ved t = {t:.4f} s")
            #break


        # Store result
        res['t'].append(t)
        gen_speed = ps.gen['GEN'].speed(x, v)
        gen_angle = ps.gen['GEN'].angle(x, v)
        frequency_hz = 50 * (1 + gen_speed[frequency_gen_idx])
        relative_angle = np.rad2deg(gen_angle[angle_gen_idx] - gen_angle[angle_reference_idx])
        mechanical_power = ps.gen['GEN'].P_m(x, v)[bergen_gen] * ps.gen['GEN'].par['S_n'][bergen_gen]
        electrical_power = ps.gen['GEN'].P_e(x, v)[bergen_gen]
        power_imbalance = mechanical_power - electrical_power
        res['voltage_5560'].append(abs(v[voltage_bus_idx]))
        res['transformer_current_ka'].append(transformer_from_current(ps, x, v))

        res['frequency_hz'].append(frequency_hz.copy())
        res['relative_angle_deg'].append(relative_angle.copy())
        res['mechanical_power_mw'].append(np.sum(mechanical_power))
        res['electrical_power_mw'].append(np.sum(electrical_power))
        #res['power_imbalance_mw'].append(np.sum(power_imbalance))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    print('Final UFLS shed power: {:.2f} MW'.format(res['ufls_shed_mw'][-1]))
    print('Peak UFLS shed power: {:.2f} MW'.format(max(res['ufls_shed_mw'])))

    time_values = np.asarray(res['t'])
    frequency_values = np.asarray(res['frequency_hz'])
    frequency_nadir = np.min(frequency_values)

    if plot_all:
        axes[0, 0].plot(time_values, frequency_values, color=col,
                        label=f'{case_label}')
        axes[0, 0].set_ylabel('Frequency [Hz]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_title('Bergen generator frequency')

        axes[0, 1].plot(time_values, res['relative_angle_deg'], color=col,
                        label=f'{case_label}') #: G5510-1 relative to G5560-1
        axes[0, 1].set_ylabel('Relative rotor angle [deg]')
        axes[0, 1].set_title('Rotor angles relative to G5560-1')

        axes[1, 0].plot(time_values, res['mechanical_power_mw'], color=col, linestyle='--',
                        label=f'{case_label} mechanical')
        axes[1, 0].plot(time_values, res['electrical_power_mw'],
                        color=col, label=f'{case_label} electrical')
        #axes[1, 0].plot(time_values, res['power_imbalance_mw'], color=col,
        #                label=f'{case_label} imbalance', linestyle=':')
        axes[1, 0].set_ylabel('Power [MW]')
        axes[1, 0].set_title('Bergen generator power response')

        axes[1, 1].plot(time_values, res['voltage_5560'], color=col,
                        label=f'{case_label}')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Voltage [p.u.]')
        axes[1, 1].set_title('Bergen bus voltages')

        axes[2, 0].plot(time_values, res['transformer_current_ka'], color=col,
                        label=f'{case_label}')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].set_ylabel('Current [kA]')
        axes[2, 0].set_title('T5510-5560 from-side current (5510 side)')
        axes[2, 1].set_visible(False)

        for axis in axes.flat:
            axis.grid(True)

    else:
        axes.plot(time_values, frequency_values, color=col, label=case_label)
        axes.set_ylabel('Frequency [Hz]')
        axes.set_xlabel('Time [s]')
        axes.set_title('Bergen generator frequency')
        axes.grid(True)
        axes.legend()

    #add_hygov_overflow_marker(axes, overflow_time)

    #for axis in axes[0, :]:
    #    axis.set_xticklabels([])

    #fig.suptitle('Bergen response after islanding')
    #fig.tight_layout()

    print('Frequency nadir for {}: {:.2f} Hz'.format(case_label, frequency_nadir))
    #plt.show()


def inertia_change(ic, case_label, col, t_end=15, UFLS_flag=False, plot_all=True):

    # Load model
    import tops.ps_models.n45_2025 as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Changing the inertia
    gen_table = model['generators']['GEN']
    h_index = gen_table[0].index('H')
    for generator_name, inertia in ic:
        gen_row = next(row for row in gen_table[1:] if row[0] == generator_name)
        gen_row[h_index] = inertia

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    overflow_time = [None]
    res = defaultdict(list)
    t_0 = time.time()

    generator_names = ps.gen['GEN'].par['name']
    bergen_gen_names = ['G5510-1', 'G5510-2', 'G5560-1']
    bergen_gen = np.array([
        int(np.flatnonzero(generator_names == name)[0])
        for name in bergen_gen_names
    ])
    gen_par = ps.gen['GEN'].par
    inertia_weights = (
        gen_par['H'][bergen_gen]
        * gen_par['S_n'][bergen_gen]
        * gen_par['N_par'][bergen_gen]
    )
    frequency_gen_idx = int(np.flatnonzero(generator_names == 'G5560-1')[0])
    angle_gen_idx = int(np.flatnonzero(generator_names == 'G5510-1')[0])
    angle_reference_idx = frequency_gen_idx
    voltage_bus_idx = int(np.flatnonzero(ps.buses['name'] == '5560')[0])

    bergen_load = np.array([31, 32])
    bergen_load_names = ps.loads['Load'].par['name'][bergen_load]

    print('Active power imbalance:', np.sum(ps.gen['GEN'].par['P'][bergen_gen]), '-', np.sum(ps.loads['Load'].par['P'][bergen_load]),
        '=', np.sum(ps.gen['GEN'].par['P'][bergen_gen]) - np.sum(ps.loads['Load'].par['P'][bergen_load]))

    island_flag = True
    #UFLS filter settings
    T_m = 0.6 # Time constant for measurement in seconds
    f_meas = ps.f_n # Nominal frequency in Hz

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Bergen is islanded
        if t > 1 and island_flag:
            island_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][50], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][7], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][14], 'disconnect')

        # Simulate next step
        step_and_detect_hygov_overflow(sol, overflow_time)
        x = sol.y
        v = sol.v
        t = sol.t

        # Under frequence load shedding
        f_actual = 50 * (1 + ps.gen['GEN'].speed(x, v)[frequency_gen_idx])
        dt = 5e-3
        f_meas = f_meas + (dt/T_m) * (f_actual - f_meas)

        apply_ufls(ps, f_meas, UFLS_flag)


        # Store result
        res['t'].append(t)
        gen_speed = ps.gen['GEN'].speed(x, v)
        gen_angle = ps.gen['GEN'].angle(x, v)
        frequency_hz = 50 * (1 + gen_speed[frequency_gen_idx])
        relative_angle = np.rad2deg(gen_angle[angle_gen_idx] - gen_angle[angle_reference_idx])
        mechanical_power = ps.gen['GEN'].P_m(x, v)[bergen_gen] * ps.gen['GEN'].par['S_n'][bergen_gen]
        electrical_power = ps.gen['GEN'].P_e(x, v)[bergen_gen]
        power_imbalance = mechanical_power - electrical_power
        res['voltage_5560'].append(abs(v[voltage_bus_idx]))
        res['transformer_current_ka'].append(transformer_from_current(ps, x, v))

        res['frequency_hz'].append(frequency_hz.copy())
        res['inertia_weighted_frequency_hz'].append(
            np.average(
                50 * (1 + gen_speed[bergen_gen]),
                weights=inertia_weights,
            )
        )
        res['relative_angle_deg'].append(relative_angle.copy())
        res['mechanical_power_mw'].append(np.sum(mechanical_power))
        res['electrical_power_mw'].append(np.sum(electrical_power))
        res['power_imbalance_mw'].append(np.sum(power_imbalance))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    time_values = np.asarray(res['t'])
    frequency_values = np.asarray(res['frequency_hz'])
    frequency_nadir = np.min(frequency_values)

    # RoCoF over the first second after islanding (t = 1 s to t = 2 s)
    t_island = 1.0
    window = (time_values >= t_island) & (time_values <= t_island + 1.0)
    rocof_frequency_values = np.asarray(res['inertia_weighted_frequency_hz'])
    rocof = np.polyfit(
        time_values[window],
        rocof_frequency_values[window],
        1,
    )[0]  # slope in Hz/s
    print('Inertia-weighted RoCoF for {}: {:.3f} Hz/s'.format(case_label, rocof))

    frequency_line, = axes[0, 0].plot(
        time_values, frequency_values, color=col, label=f'{case_label}: G5560-1')
    axes[0, 0].axhline(frequency_nadir, color=col, linestyle='--',
                        label=f'{case_label} nadir: {frequency_nadir:.2f} Hz')
    axes[0, 0].set_ylabel('Frequency [Hz]')
    axes[0, 0].set_title('Bergen generator frequency')

    if plot_all:
        axes[0, 1].plot(time_values, res['relative_angle_deg'], color=col,
                        label=f'{case_label}: G5510-1 relative to G5560-1')
        axes[0, 1].set_ylabel('Relative rotor angle [deg]')
        axes[0, 1].set_title('Rotor angles relative to G5560-1')

        axes[1, 0].plot(time_values, res['mechanical_power_mw'], color=col, linestyle='--',
                        label=f'{case_label} mechanical')
        axes[1, 0].plot(time_values, res['electrical_power_mw'],
                        color=col, label=f'{case_label} electrical')
        axes[1, 0].plot(time_values, res['power_imbalance_mw'], color=col,
                        label=f'{case_label} imbalance', linestyle=':')
        axes[1, 0].set_ylabel('Power [MW]')
        axes[1, 0].set_title('Bergen generator power response')

        axes[1, 1].plot(time_values, res['voltage_5560'], color=col,
                        label=f'{case_label}: bus 5560')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Voltage [p.u.]')
        axes[1, 1].set_title('Bergen bus voltages')

        axes[2, 0].plot(time_values, res['transformer_current_ka'], color=col,
                        label=f'{case_label}: T5510-5560')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].set_ylabel('Current [kA]')
        axes[2, 0].set_title('T5510-5560 from-side current (5510 side)')
        axes[2, 1].set_visible(False)
    else:
        for axis in axes.flat:
            if axis is not axes[0, 0]:
                axis.set_visible(False)

    #add_hygov_overflow_marker(axes, overflow_time)

    for axis in axes.flat:
        axis.grid(True)


    fig.suptitle('Bergen response after islanding')
    fig.tight_layout()


    # plotting eigenvalues
    """
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    figu, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
    """


def T_m_change(T_m, case_label, col, t_end=15, UFLS_flag=False, plot_all=True):
    island_flag = True

    # Load model
    import tops.ps_models.n45_2025 as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Changing the time constant is done

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    overflow_time = [None]
    res = defaultdict(list)
    t_0 = time.time()

    generator_names = ps.gen['GEN'].par['name']
    bergen_gen_names = ['G5510-1', 'G5510-2', 'G5560-1']
    bergen_gen = np.array([
        int(np.flatnonzero(generator_names == name)[0])
        for name in bergen_gen_names
    ])
    frequency_gen_idx = int(np.flatnonzero(generator_names == 'G5560-1')[0])
    angle_gen_idx = int(np.flatnonzero(generator_names == 'G5510-1')[0])
    angle_reference_idx = frequency_gen_idx
    voltage_bus_idx = int(np.flatnonzero(ps.buses['name'] == '5560')[0])

    bergen_load = np.array([31, 32])
    bergen_load_names = ps.loads['Load'].par['name'][bergen_load]

    print('Active power imbalance:', np.sum(ps.gen['GEN'].par['P'][bergen_gen]), '-', np.sum(ps.loads['Load'].par['P'][bergen_load]),
        '=', np.sum(ps.gen['GEN'].par['P'][bergen_gen]) - np.sum(ps.loads['Load'].par['P'][bergen_load]))


    #UFLS filter settings
    #T_m = 0.05 # Time constant for measurement in seconds
    f_meas = ps.f_n # Nominal frequency in Hz

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))


        # Bergen is islanded
        if t > 1 and island_flag:
            island_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][50], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][7], 'disconnect')
            ps.trafos['Trafo'].event(ps, ps.trafos['Trafo'].par['name'][14], 'disconnect')

        # Simulate next step
        step_and_detect_hygov_overflow(sol, overflow_time)
        x = sol.y
        v = sol.v
        t = sol.t

        # Under frequence load shedding
        f_actual = 50 * (1 + ps.gen['GEN'].speed(x, v)[frequency_gen_idx])
        dt = 5e-3
        f_meas = f_meas + (dt/T_m) * (f_actual - f_meas)

        apply_ufls(ps, f_meas, UFLS_flag)


        # Store result
        res['t'].append(t)
        gen_speed = ps.gen['GEN'].speed(x, v)
        gen_angle = ps.gen['GEN'].angle(x, v)
        frequency_hz = 50 * (1 + gen_speed[frequency_gen_idx])
        relative_angle = np.rad2deg(gen_angle[angle_gen_idx] - gen_angle[angle_reference_idx])
        mechanical_power = ps.gen['GEN'].P_m(x, v)[bergen_gen] * ps.gen['GEN'].par['S_n'][bergen_gen]
        electrical_power = ps.gen['GEN'].P_e(x, v)[bergen_gen]
        power_imbalance = mechanical_power - electrical_power
        res['voltage_5560'].append(abs(v[voltage_bus_idx]))
        res['transformer_current_ka'].append(transformer_from_current(ps, x, v))

        res['frequency_hz'].append(frequency_hz.copy())
        res['relative_angle_deg'].append(relative_angle.copy())
        res['mechanical_power_mw'].append(np.sum(mechanical_power))
        res['electrical_power_mw'].append(np.sum(electrical_power))
        res['power_imbalance_mw'].append(np.sum(power_imbalance))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    time_values = np.asarray(res['t'])
    frequency_values = np.asarray(res['frequency_hz'])
    frequency_nadir = np.min(frequency_values)


    if plot_all:
        frequency_line, = axes[0, 0].plot(
                time_values, frequency_values, color=col, label=f'{case_label}: G5560-1')
        axes[0, 0].axhline(frequency_nadir, color=col, linestyle='--',
                            label=f'{case_label} nadir: {frequency_nadir:.2f} Hz')
        axes[0, 0].set_ylabel('Frequency [Hz]')
        axes[0, 0].set_title('Bergen generator frequency')
        axes[0, 1].plot(time_values, res['relative_angle_deg'], color=col,
                        label=f'{case_label}: G5510-1 relative to G5560-1')
        axes[0, 1].set_ylabel('Relative rotor angle [deg]')
        axes[0, 1].set_title('Rotor angles relative to G5560-1')

        axes[1, 0].plot(time_values, res['mechanical_power_mw'], color=col, linestyle='--',
                        label=f'{case_label} mechanical')
        axes[1, 0].plot(time_values, res['electrical_power_mw'],
                        color=col, label=f'{case_label} electrical')
        axes[1, 0].plot(time_values, res['power_imbalance_mw'], color=col,
                        label=f'{case_label} imbalance', linestyle=':')
        axes[1, 0].set_ylabel('Power [MW]')
        axes[1, 0].set_title('Bergen generator power response')

        axes[1, 1].plot(time_values, res['voltage_5560'], color=col,
                        label=f'{case_label}: bus 5560')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Voltage [p.u.]')
        axes[1, 1].set_title('Bergen bus voltages')

        axes[2, 0].plot(time_values, res['transformer_current_ka'], color=col,
                        label=f'{case_label}: T5510-5560')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].set_ylabel('Current [kA]')
        axes[2, 0].set_title('T5510-5560 from-side current (5510 side)')
        axes[2, 1].set_visible(False)

        for axis in axes.flat:
            axis.grid(True)
    else:
        axes.plot(time_values, frequency_values, color=col, label=case_label)
        axes.set_ylabel('Frequency [Hz]')
        axes.set_xlabel('Time [s]')
        axes.set_title('Bergen generator frequency')
        axes.grid(True)
        """
        axes_right = axes.twinx()
        axes_right.plot(time_values, res['power_imbalance_mw'], color=col, linestyle='--', label=f'{case_label} Imbalance')
        axes_right.set_ylabel('Power [MW]')
        lines_left, labels_left = axes.get_legend_handles_labels()
        lines_right, labels_right = axes_right.get_legend_handles_labels()
        axes.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right')
        axes_right.set_ylim([-600, 100])
        """
        axes.legend()

    #add_hygov_overflow_marker(axes, overflow_time)


    fig.suptitle('Bergen response after islanding')
    fig.tight_layout()



if __name__ == '__main__':
    variable = 'lc'
    t_end = 65

    C = 0

    if variable == 'general':
        general(15)

    if variable == 'lc':
        lc = []
        
        #fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
        fig, axes = plt.subplots(1, 1, figsize=(6, 4.5), sharex=True)
        """
        for i in range(1200, 1451, 50):

            lc.append([['L5560-1', i],
                       ['L5560-2', i]])
            load_change(lc[C], f'P_load = {i}', f'C{C}', t_end, UFLS_flag=True, plot_all=False)
            print(lc[C], f'P_load = {i}', f'C{C}', t_end, C)
            C +=1
        """

        load_change([['L5560-1', 1117.1178933215733], ['L5560-2', 1117.1178933215733]], f'$\Delta$P = -546 MW', f'C{0}', t_end, UFLS_flag=False, plot_all=False)
        load_change([['L5560-1', 1185], ['L5560-2', 1185]], f'$\Delta$P = -681 MW', f'C{1}', t_end, UFLS_flag=False, plot_all=False)
        load_change([['L5560-1', 1350], ['L5560-2', 1350]], f'$\Delta$P = -1011 MW', f'C{2}', t_end, UFLS_flag=True, plot_all=False)

        """
        for axis in axes.flat:
            axis.legend()
        """
        plt.show()  

    if variable == 'ic':
        # inertia change
        ic = []

        fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

        for i in range(40, 41, 10):

            ic.append([['G5560-1', i*0.1],
                       ['G5510-1', i*0.1],
                       ['G5510-2', i*0.1]])
            inertia_change(ic[C], f'H = {i*0.1}', f'C{C}', t_end, UFLS_flag=True)
            print(ic[C], f'H = {i*0.1}', f'C{C}', t_end, C)
            C +=1
        
        for axis in axes.flat:
            axis.legend()
        plt.show()

    if variable == 'T_m':
        li = [6, 15]

        #fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
        fig, axes = plt.subplots(1, 1, figsize=(3.5, 2.2), sharex=True)

        for i in li:
            a = round(i*0.1, 2)
            T_m_change(a, f'T_m = {a}', f'C{C}', t_end, UFLS_flag=True, plot_all=False)
            print(a, f'T_m = {a}', f'C{C}', t_end)
            C +=1


        
        #for axis in axes.flat:
        #    axis.legend()
        plt.show()