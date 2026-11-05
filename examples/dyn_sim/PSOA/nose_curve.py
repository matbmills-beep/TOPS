"""Trace an approximate N45 voltage nose curve at bus 5560 with load flow."""

import contextlib
import copy
import io

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.ps_models.n45_2025 as n45_model


LOAD_BUS = '5560'
GENERATOR_NAME = 'G5560-1'
INITIAL_STEP = 0.05
MINIMUM_STEP_MW = 1.0
MAXIMUM_LOAD_FACTOR = 20.0
Q_MAX_FRACTION_OF_RATED_S = 0.6


def generator_limits(model):
    generator_table = model['generators']['GEN']
    header = generator_table[0]
    n_par_index = header.index('N_par') if 'N_par' in header else None
    specs = {}
    for row in generator_table[1:]:
        n_par = float(row[n_par_index]) if n_par_index is not None else 1.0
        specs[row[header.index('name')]] = {
            'bus': row[header.index('bus')],
            'p': float(row[header.index('P')]) * n_par,
            'q_max': Q_MAX_FRACTION_OF_RATED_S * float(row[header.index('S_n')]) * n_par,
        }
    return specs


def model_with_q_limited_generators(base_model, load_factor, base_bus_loads,
                                   generator_specs, fixed_generators):
    model = copy.deepcopy(base_model)
    load_table = model['loads']
    name_index = load_table[0].index('name')
    bus_index = load_table[0].index('bus')
    p_index = load_table[0].index('P')
    q_index = load_table[0].index('Q')
    base_load_by_name = {
        name: (p, q) for name, p, q in base_bus_loads
    }
    for row in load_table[1:]:
        if row[bus_index] == LOAD_BUS and row[name_index] in base_load_by_name:
            base_p, base_q = base_load_by_name[row[name_index]]
            row[p_index] = base_p * load_factor
            row[q_index] = base_q * load_factor

    generator_table = model['generators']['GEN']
    generator_name_index = generator_table[0].index('name')
    model['generators']['GEN'] = [generator_table[0]] + [
        row for row in generator_table[1:]
        if row[generator_name_index] not in fixed_generators
    ]
    for category in ('gov', 'avr', 'pss'):
        for controller_table in model.get(category, {}).values():
            generator_index = controller_table[0].index('gen')
            controller_table[1:] = [
                row for row in controller_table[1:]
                if row[generator_index] not in fixed_generators
            ]
    for generator_name in fixed_generators:
        spec = generator_specs[generator_name]
        load_table.append([
            'Q-limited ' + generator_name,
            spec['bus'],
            -spec['p'],
            -spec['q_max'],
            'Z',
        ])
    return model


def solve_at_load_factor(base_model, load_factor, base_bus_loads, generator_specs):
    fixed_generators = set()
    slack_bus = base_model.get('slack_bus')

    for _ in range(len(generator_specs) + 1):
        model = model_with_q_limited_generators(
            base_model, load_factor, base_bus_loads, generator_specs,
            fixed_generators)
        system = dps.PowerSystemModel(model=model)
        system.pf_max_it = 60
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                converged = system.power_flow()
        except np.linalg.LinAlgError:
            return False, None, fixed_generators

        if not converged:
            return False, None, fixed_generators

        generator_model = system.gen['GEN']
        generator_solution = system.load_flow_soln[generator_model]
        q_by_bus = {}
        generators_by_bus = {}
        for index, generator_name in enumerate(generator_model.par['name']):
            bus = generator_model.par['bus'][index]
            q_by_bus[bus] = q_by_bus.get(bus, 0.0) + generator_solution.imag[index]
            generators_by_bus.setdefault(bus, []).append(generator_name)

        newly_limited = set()
        for bus, required_q in q_by_bus.items():
            if bus == slack_bus:
                continue
            q_max = sum(generator_specs[name]['q_max']
                        for name in generators_by_bus[bus])
            if required_q > q_max + 1e-6:
                newly_limited.update(generators_by_bus[bus])

        if not newly_limited:
            bus_index = int(np.flatnonzero(system.buses['name'] == LOAD_BUS)[0])
            return True, abs(system.v_0[bus_index]), fixed_generators
        fixed_generators.update(newly_limited)

    raise RuntimeError('Q-limit active-set iteration did not settle.')


def run():
    base_model = n45_model.load()
    generator_specs = generator_limits(base_model)
    base_loads = base_model['loads']
    name_index = base_loads[0].index('name')
    bus_index = base_loads[0].index('bus')
    p_index = base_loads[0].index('P')
    q_index = base_loads[0].index('Q')
    bus_loads = [
        row for row in base_loads[1:]
        if row[bus_index] == LOAD_BUS
    ]
    base_bus_loads = [
        (row[name_index], float(row[p_index]), float(row[q_index]))
        for row in bus_loads
    ]
    bus_load_names = [name for name, _, _ in base_bus_loads]
    base_bus_p = sum(p for _, p, _ in base_bus_loads)
    base_bus_q = sum(q for _, _, q in base_bus_loads)
    converged, voltage, zero_load_limited_generators = solve_at_load_factor(
        base_model, 0.0, base_bus_loads, generator_specs)
    if not converged:
        raise RuntimeError('The N45 power flow did not converge at zero bus load.')

    basecase_converged, basecase_voltage, basecase_limited_buses = solve_at_load_factor(
        base_model, 1.0, base_bus_loads, generator_specs)
    if not basecase_converged:
        raise RuntimeError('The N45 base-case equivalent power flow did not converge.')

    load_factors = [0.0]
    bus_voltages = [voltage]
    current_factor = 0.0
    step = INITIAL_STEP
    first_failed_factor = None
    limited_generators_seen = set(zero_load_limited_generators)
    limited_generators_seen.update(basecase_limited_buses)
    while (step * base_bus_p >= MINIMUM_STEP_MW
           and current_factor < MAXIMUM_LOAD_FACTOR):
        trial_factor = min(current_factor + step, MAXIMUM_LOAD_FACTOR)
        converged, voltage, limited_generators = solve_at_load_factor(
            base_model, trial_factor, base_bus_loads, generator_specs)
        limited_generators_seen.update(limited_generators)

        if converged:
            current_factor = trial_factor
            load_factors.append(current_factor)
            bus_voltages.append(voltage)
        else:
            first_failed_factor = trial_factor
            step /= 2

    critical_bus_p = base_bus_p * current_factor
    critical_bus_q = base_bus_q * current_factor
    print('Last converged point near the voltage-stability limit:')
    print('  Load bus: {} (loads {})'.format(
        LOAD_BUS, ', '.join(bus_load_names)))
    print('  All loads on this bus were scaled together from zero.')
    print('  Total load at bus {}: P = {:.2f} MW, Q = {:.2f} MVAr'.format(
        LOAD_BUS, critical_bus_p, critical_bus_q))
    print('  |V| = {:.4f} pu'.format(bus_voltages[-1]))
    print('  Common multiplier for all loads at bus {} = {:.5f}'.format(
        LOAD_BUS, current_factor))
    if limited_generators_seen:
        print('  PV generator limits are Qmax = {:.0%} of each generator rated S.'.format(
            Q_MAX_FRACTION_OF_RATED_S))
        limited_buses = sorted({
            generator_specs[name]['bus'] for name in limited_generators_seen
        })
        print('  PV buses converted to PQ at Qmax during the sweep: {}'.format(
            ', '.join(limited_buses)))
    if first_failed_factor is None:
        print('  Sweep reached its maximum load factor without finding the NR limit.')
    else:
        failed_bus_p = base_bus_p * first_failed_factor
        print('  First failed solve total bus load P = {:.2f} MW'.format(failed_bus_p))
        print('  Last-converged/first-failed bracket = {:.2f} MW'.format(
            failed_bus_p - critical_bus_p))
        print('  This is an approximate critical point from NR convergence, not a '
              'formal continuation-power-flow solution.')
    print('  PV generators at non-slack buses are fixed at Qmax and converted to '
          'PQ when their bus Q requirement exceeds aggregate capability.')
    print('  Nominal-load point with generator Q limits: P = {:.2f} MW, Q = {:.4f} MVAr, '
          '|V| = {:.4f} pu'.format(
              base_bus_p, base_bus_q, basecase_voltage))

    total_bus_active_loads = base_bus_p * np.asarray(load_factors)
    plt.plot(total_bus_active_loads, bus_voltages, 'o-', markersize=3,
             label='Load-flow solutions')
    plt.scatter([base_bus_p], [basecase_voltage], color='C1', marker='s',
                zorder=4, label='Nominal load (with Q limits)')
    plt.scatter([critical_bus_p], [bus_voltages[-1]], color='red', zorder=3,
                label='Last converged point')
    plt.xlabel('Total active load connected to bus 5560 [MW]')
    plt.ylabel('Bus 5560 voltage magnitude [p.u.]')
    plt.title('N45 voltage nose curve at bus 5560')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()