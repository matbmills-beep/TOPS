import importlib
import csv

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.ps_models.n45_2025 as full_model_data
import tops.ps_models.n45_2025_bergen as bergen_model_data


BERGEN_BUSES = {'5510', '5560'}
slack_bus = '5510'

# --- distributed slack additions ---
USE_DISTRIBUTED_SLACK = True
PARTICIPATING_GENERATORS = ['G5510-1', 'G5510-2', 'G5560-1']
ANGLE_REF_BUS = '5510'


def make_system(model_data):
    importlib.reload(model_data)
    ps = dps.PowerSystemModel(model=model_data.load())
    ps.setup()
    ps.slack_bus = slack_bus
    return ps


def bus_index(ps, bus_name):
    return int(np.flatnonzero(ps.buses['name'] == bus_name)[0])


def generator_solution(ps):
    solution = ps.load_flow_soln[ps.gen['GEN']]
    return solution.real, solution.imag


def load_totals(ps):
    return np.sum(ps.loads['Load'].par['P']), np.sum(ps.loads['Load'].par['Q'])


def generator_totals(ps):
    active, reactive = generator_solution(ps)
    total_active = np.sum(active)
    total_reactive = np.sum(reactive)
    for vsc in getattr(ps, 'vsc', {}).values():
        total_active += np.sum(vsc.par['p_ref'] * vsc.par['S_n'])
        total_reactive += np.sum(vsc.par['q_ref'] * vsc.par['S_n'])
    return total_active, total_reactive


def bergen_totals(ps):
    bergen_bus_names = list(BERGEN_BUSES)
    generator_mask = np.isin(ps.gen['GEN'].par['bus'], bergen_bus_names)
    load_mask = np.isin(ps.loads['Load'].par['bus'], bergen_bus_names)
    generator_p, generator_q = generator_solution(ps)
    return (
        np.sum(generator_p[generator_mask]),
        np.sum(generator_q[generator_mask]),
        np.sum(ps.loads['Load'].par['P'][load_mask]),
        np.sum(ps.loads['Load'].par['Q'][load_mask]),
    )


def branch_flow(branch, index, v, s_n):
    p_from = float(branch.p_from(None, v)[index]) * s_n
    q_from = float(branch.q_from(None, v)[index]) * s_n
    p_to = float(branch.p_to(None, v)[index]) * s_n
    q_to = float(branch.q_to(None, v)[index]) * s_n
    return p_from, q_from, p_to, q_to


def boundary_flows(ps):
    flows = []

    line = getattr(ps, 'lines', {}).get('Line')
    if line is not None:
        for index, name in enumerate(line.par['name']):
            from_bus = line.par['from_bus'][index]
            to_bus = line.par['to_bus'][index]
            if (from_bus in BERGEN_BUSES) != (to_bus in BERGEN_BUSES):
                flows.append(('Line', name, from_bus, to_bus, index))

    trafo = getattr(ps, 'trafos', {}).get('Trafo')
    if trafo is not None:
        for index, name in enumerate(trafo.par['name']):
            from_bus = trafo.par['from_bus'][index]
            to_bus = trafo.par['to_bus'][index]
            if (from_bus in BERGEN_BUSES) != (to_bus in BERGEN_BUSES):
                flows.append(('Trafo', name, from_bus, to_bus, index))

    return flows


def signed_bergen_flow(ps, flow, v):
    kind, name, from_bus, to_bus, index = flow
    branch = ps.lines['Line'] if kind == 'Line' else ps.trafos['Trafo']
    p_from, q_from, p_to, q_to = branch_flow(branch, index, v, ps.s_n)
    if from_bus in BERGEN_BUSES:
        return p_from, q_from
    return p_to, q_to


def print_case(title, ps, converged, bus_names=None, generator_names=None, transformer_names=None):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)
    print('Power-flow converged:', converged)

    print('\nBus voltages:')
    buses_to_print = ps.buses['name'] if bus_names is None else bus_names
    for bus_name in buses_to_print:
        voltage = ps.v_0[bus_index(ps, bus_name)]
        print('  {:>6}: {: .6f} pu, {: .3f} deg'.format(
            bus_name, abs(voltage), np.rad2deg(np.angle(voltage))))

    gen_p, gen_q = generator_solution(ps)
    print('\nGenerator output:')
    generators_to_print = ps.gen['GEN'].par['name'] if generator_names is None else generator_names
    for name in generators_to_print:
        generator_index = int(np.flatnonzero(ps.gen['GEN'].par['name'] == name)[0])
        p = gen_p[generator_index]
        q = gen_q[generator_index]
        print('  {:>10}: P = {: .3f} MW, Q = {: .3f} MVAr'.format(name, p, q))

    load_p, load_q = load_totals(ps)
    total_gen_p, total_gen_q = generator_totals(ps)
    
    print('\nSystem totals:')
    print('  Total generation: P = {: .3f} MW, Q = {: .3f} MVAr'.format(total_gen_p, total_gen_q))
    print('  Load:       P = {: .3f} MW, Q = {: .3f} MVAr'.format(load_p, load_q))
    print('  Total generation minus load: P = {: .3f} MW, Q = {: .3f} MVAr'.format(
        total_gen_p - load_p, total_gen_q - load_q))

    bergen_gen_p, bergen_gen_q, bergen_load_p, bergen_load_q = bergen_totals(ps)
    print('\nBergen totals:')
    print('  Generation: P = {: .3f} MW, Q = {: .3f} MVAr'.format(bergen_gen_p, bergen_gen_q))
    print('  Load:       P = {: .3f} MW, Q = {: .3f} MVAr'.format(bergen_load_p, bergen_load_q))
    print('  Balance:    P = {: .3f} MW, Q = {: .3f} MVAr'.format(
        bergen_gen_p - bergen_load_p, bergen_gen_q - bergen_load_q))

    flows = boundary_flows(ps)
    if flows:
        print('\nPower flow at Bergen boundary (positive means leaving Bergen):')
        boundary_p = 0.0
        boundary_q = 0.0
        for flow in flows:
            kind, name, from_bus, to_bus, index = flow
            p, q = signed_bergen_flow(ps, flow, ps.v_0)
            boundary_p += p
            boundary_q += q
            print('  {:>5} {:>12} ({:>4} -> {:>4}): P = {: .3f} MW, Q = {: .3f} MVAr'.format(
                kind, name, from_bus, to_bus, p, q))
        print('  Net boundary export: P = {: .3f} MW, Q = {: .3f} MVAr'.format(
            boundary_p, boundary_q))
        print('  Net boundary import: P = {: .3f} MW, Q = {: .3f} MVAr'.format(
            -boundary_p, -boundary_q))
    else:
        print('\nBergen boundary flow: no external branches in this model.')

    if getattr(ps, 'trafos', {}).get('Trafo') is not None and len(ps.trafos['Trafo'].par) > 0:
        print('\nInternal transformer flows:')
        transformers_to_print = ps.trafos['Trafo'].par['name'] if transformer_names is None else transformer_names
        for name in transformers_to_print:
            index = int(np.flatnonzero(ps.trafos['Trafo'].par['name'] == name)[0])
            p_from, q_from, p_to, q_to = branch_flow(ps.trafos['Trafo'], index, ps.v_0, ps.s_n)
            print('  {:>12}: from P = {: .3f} MW, Q = {: .3f} MVAr; '
                  'to P = {: .3f} MW, Q = {: .3f} MVAr'.format(
                      name, p_from, q_from, p_to, q_to))


# --- distributed slack additions ---
def solve_distributed_slack(ps):
    if USE_DISTRIBUTED_SLACK:
        converged = ps.power_flow_distributed_slack(
            participating_gen_names=PARTICIPATING_GENERATORS,
            angle_ref_bus=ANGLE_REF_BUS,
        )
        return converged
    return ps.power_flow()


def print_distributed_slack_info(ps):
    if not USE_DISTRIBUTED_SLACK:
        return
    print('\nDistributed-slack results:')
    print('  dP = {: .6f} MW'.format(ps.dP_distributed_slack))
    print('  Participation factors:')
    for gen_name in PARTICIPATING_GENERATORS:
        alpha = ps.distributed_slack_participation.get(gen_name, 0.0)
        print('    {:>10}: alpha = {:.6f}'.format(gen_name, alpha))


# --- CSV export additions ---
def collect_case_data(title, ps, converged, bus_names=None, generator_names=None, transformer_names=None):
    """Gather exactly the same numbers print_case() prints, as plain data.
 
    Returns a dict that create_csv_file() knows how to lay out. Keeping this
    separate from print_case() means the printed report and the CSV can never
    silently drift apart in terms of *which* numbers they show, only how
    they're displayed.
    """
    data = {'title': title, 'converged': converged}
 
    # Bus voltages
    buses_to_use = ps.buses['name'] if bus_names is None else bus_names
    bus_voltages = {}
    for bus_name in buses_to_use:
        voltage = ps.v_0[bus_index(ps, bus_name)]
        bus_voltages[str(bus_name)] = [abs(voltage), np.rad2deg(np.angle(voltage))]
    data['bus_voltages'] = bus_voltages
 
    # Generator output
    gen_p, gen_q = generator_solution(ps)
    generators_to_use = ps.gen['GEN'].par['name'] if generator_names is None else generator_names
    generator_output = {}
    for name in generators_to_use:
        generator_index = int(np.flatnonzero(ps.gen['GEN'].par['name'] == name)[0])
        generator_output[str(name)] = [gen_p[generator_index], gen_q[generator_index]]
    data['generator_output'] = generator_output
 
    # System totals
    load_p, load_q = load_totals(ps)
    total_gen_p, total_gen_q = generator_totals(ps)
    data['system_totals'] = {
        'Total generation': [total_gen_p, total_gen_q],
        'Load': [load_p, load_q],
        'Total generation minus load': [total_gen_p - load_p, total_gen_q - load_q],
    }
 
    # Bergen totals
    bergen_gen_p, bergen_gen_q, bergen_load_p, bergen_load_q = bergen_totals(ps)
    data['bergen_totals'] = {
        'Generation': [bergen_gen_p, bergen_gen_q],
        'Load': [bergen_load_p, bergen_load_q],
        'Balance': [bergen_gen_p - bergen_load_p, bergen_gen_q - bergen_load_q],
    }
 
    # Boundary flows
    flows = boundary_flows(ps)
    boundary = {}
    boundary_p_total = 0.0
    boundary_q_total = 0.0
    for flow in flows:
        kind, name, from_bus, to_bus, index = flow
        p, q = signed_bergen_flow(ps, flow, ps.v_0)
        boundary_p_total += p
        boundary_q_total += q
        row_label = '{} {} ({} -> {})'.format(kind, name, from_bus, to_bus)
        boundary[row_label] = [p, q]
    if flows:
        boundary['Net boundary export'] = [boundary_p_total, boundary_q_total]
        boundary['Net boundary import'] = [-boundary_p_total, -boundary_q_total]
    data['boundary_flows'] = boundary
 
    # Internal transformer flows
    transformer_flows = {}
    if getattr(ps, 'trafos', {}).get('Trafo') is not None and len(ps.trafos['Trafo'].par) > 0:
        transformers_to_use = ps.trafos['Trafo'].par['name'] if transformer_names is None else transformer_names
        for name in transformers_to_use:
            index = int(np.flatnonzero(ps.trafos['Trafo'].par['name'] == name)[0])
            p_from, q_from, p_to, q_to = branch_flow(ps.trafos['Trafo'], index, ps.v_0, ps.s_n)
            transformer_flows[str(name)] = [p_from, q_from, p_to, q_to]
    data['transformer_flows'] = transformer_flows
 
    return data
 
 
# (section label, data key, sub-column headers)
_CSV_SECTIONS = [
    ('Bus voltages', 'bus_voltages', ['v [pu]', 'angle [deg]']),
    ('Generator output', 'generator_output', ['P [MW]', 'Q [MVAr]']),
    ('System totals', 'system_totals', ['P [MW]', 'Q [MVAr]']),
    ('Bergen totals', 'bergen_totals', ['P [MW]', 'Q [MVAr]']),
    ('Boundary flows (Bergen)', 'boundary_flows', ['P [MW]', 'Q [MVAr]']),
    ('Internal transformer flows', 'transformer_flows',
     ['P_from [MW]', 'Q_from [MVAr]', 'P_to [MW]', 'Q_to [MVAr]']),
]
 
 
def create_csv_file(cases, filename='power_flow_results.csv'):
    """Write all cases side by side into a single CSV sheet.
 
    `cases` is a list of dicts produced by collect_case_data(), one per case,
    in the order they should appear left-to-right (matching the order they
    were printed in run()).
 
    Layout: column A holds the section label, column B holds the row label
    (bus/generator/transformer name, or totals row name) shared across all
    cases. Each case then gets its own fixed-width block of value columns,
    headed by the case title, with a blank column separating cases -
    mirroring the block-per-case style of the reference sheet.
    """
    block_width = max(len(headers) for _, _, headers in _CSV_SECTIONS)
    gap = 1
    n_cases = len(cases)
 
    def case_start_col(i):
        return 2 + i * (block_width + gap)  # columns 0,1 = section/row labels
 
    total_cols = case_start_col(n_cases - 1) + block_width if n_cases else 2
 
    def blank_row():
        return ['' for _ in range(total_cols)]
 
    def fmt(value):
        return '' if value is None else '{:.6f}'.format(value)
 
    rows = []
 
    # Case titles
    row = blank_row()
    for i, case in enumerate(cases):
        row[case_start_col(i)] = case['title']
    rows.append(row)
 
    # Converged flag
    row = blank_row()
    for i, case in enumerate(cases):
        row[case_start_col(i)] = 'Power-flow converged: {}'.format(case['converged'])
    rows.append(row)
 
    rows.append(blank_row())
 
    for section_label, key, headers in _CSV_SECTIONS:
        # Union of row labels across all cases, preserving first-seen order,
        # so e.g. a contingency case missing a line just leaves that row blank.
        row_order = []
        seen = set()
        for case in cases:
            for row_label in case[key].keys():
                if row_label not in seen:
                    seen.add(row_label)
                    row_order.append(row_label)
 
        if not row_order:
            continue
 
        header_row = blank_row()
        for i in range(n_cases):
            start = case_start_col(i)
            for j, header in enumerate(headers):
                header_row[start + j] = header
        rows.append(header_row)
 
        for row_number, row_label in enumerate(row_order):
            data_row = blank_row()
            if row_number == 0:
                data_row[0] = section_label
            data_row[1] = row_label
            for i, case in enumerate(cases):
                values = case[key].get(row_label)
                if values is not None:
                    start = case_start_col(i)
                    for j, value in enumerate(values):
                        data_row[start + j] = fmt(value)
            rows.append(data_row)
 
        rows.append(blank_row())
 
    with open(filename, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
 
    print('Wrote {} ({} case(s))'.format(filename, n_cases))
 
 




def run():
    bus_names = ['5510', '5560']
    generator_names = ['G5510-1', 'G5510-2', 'G5560-1']
    transformer_names = ['T5550-5560', 'T5260-5510']
    line_names = ['L5510-5520']

    cases = []

    
    full_ps = make_system(full_model_data)
    full_before_converged = solve_distributed_slack(full_ps)
    print_case('Full n45_2025 before islanding', full_ps, full_before_converged, bus_names, 
               generator_names, transformer_names)
    cases.append(collect_case_data('Full n45_2025 before islanding', full_ps, full_before_converged,
                                    bus_names, generator_names, transformer_names))

    bergen_ps = make_system(bergen_model_data)
    bergen_converged = solve_distributed_slack(bergen_ps)
    print_case('n45_2025_bergen after islanding', bergen_ps, bergen_converged)
    print_distributed_slack_info(bergen_ps)
    cases.append(collect_case_data('n45_2025_bergen after islanding', bergen_ps, bergen_converged))

    # Contingency analysis
    """
    for connection in line_names:
        contingency_ps = make_system(full_model_data)
        contingency_ps.power_flow()
        contingency_ps.lines['Line'].event(contingency_ps, connection, 'disconnect')
        contingency_ps.power_flow_ready = False
        contingency_converged = solve_distributed_slack(contingency_ps)
        print_case('Contingency analysis, outage of line ' + connection,
                   contingency_ps, contingency_converged,
                   bus_names, generator_names, transformer_names)
        cases.append(collect_case_data('Contingency analysis, outage of line ' + connection, contingency_ps, contingency_converged,
                                        bus_names, generator_names, transformer_names))

    for connection in transformer_names:
        contingency_ps = make_system(full_model_data)
        contingency_ps.power_flow()
        contingency_ps.trafos['Trafo'].event(contingency_ps, connection, 'disconnect')
        contingency_ps.power_flow_ready = False
        contingency_converged = solve_distributed_slack(contingency_ps)
        print_case('Contingency analysis, outage of transformer ' + connection,
                   contingency_ps, contingency_converged,
                   bus_names, generator_names, transformer_names)
        cases.append(collect_case_data('Contingency analysis, outage of transformer ' + connection, contingency_ps, contingency_converged,
                                        bus_names, generator_names, transformer_names))
    """

    contingency_ps = make_system(full_model_data)
    contingency_ps.power_flow()
    contingency_ps.trafos['Trafo'].event(contingency_ps, 'T5550-5560', 'disconnect')
    contingency_ps.lines['Line'].event(contingency_ps, 'L5510-5520', 'disconnect')
    contingency_ps.power_flow_ready = False
    contingency_converged = solve_distributed_slack(contingency_ps)
    print_case('Contingency analysis, outage of transformer ' + 'T5550-5560 and L5510-5520',
                contingency_ps, contingency_converged,
                bus_names, generator_names, transformer_names)
    cases.append(collect_case_data('Contingency analysis, outage of T5550-5560 and L5510-5520', contingency_ps, contingency_converged,
                                    bus_names, generator_names, transformer_names))

    create_csv_file(cases)
    
    bus_names = ['5510', '5560']
    full_before_voltage = [abs(full_ps.v_0[bus_index(full_ps, name)]) for name in bus_names]
    bergen_voltage = [abs(bergen_ps.v_0[bus_index(bergen_ps, name)]) for name in bus_names]

    x = np.arange(len(bus_names))
    width = 0.35
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, full_before_voltage, width, label='Full system before islanding')
    axis.bar(x + width / 2, bergen_voltage, width, label='Two-bus Bergen model')
    axis.set_xticks(x, bus_names)
    axis.set_ylabel('Voltage magnitude [p.u.]')
    axis.set_title('Static Bergen voltage comparison')
    axis.legend()
    axis.grid(axis='y')
    fig.tight_layout()
    #plt.show()


if __name__ == '__main__':
    run()
