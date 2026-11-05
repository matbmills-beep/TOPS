import numpy as np
from tops.solvers import Euler, ModifiedEuler, SimpleRK4


def newton_rhapson_power_flow(y_bus, v_0, p_sum_bus, q_sum_bus, bus_types, tol, pf_max_it):

    n_bus = len(bus_types)

    # Indices of PV, PQ, PV+PQ and SL-buses
    pv_idx = np.where(bus_types == 'PV')[0]
    pq_idx = np.where(bus_types == 'PQ')[0]
    pvpq_idx = np.concatenate([pv_idx, pq_idx])

    n_gen_bus = len(pv_idx) + 1

    # Map x to angles and voltages
    idx_phi = range(n_bus - 1)
    idx_v = range(n_bus - 1, n_bus - 1 + len(pq_idx))

    def x_to_v(x):
        phi = np.zeros(n_bus)
        phi[pvpq_idx] = x[idx_phi]
        v = v_0.copy()
        v[pq_idx] = x[idx_v]
        v_ph = v * np.exp(1j * phi)
        return v_ph

    # Initial guess: Flat start
    phi_0 = np.zeros(n_bus)

    x0 = np.zeros(2 * (n_bus - 1) - (n_gen_bus - 1))
    x0[idx_phi] = phi_0[pvpq_idx]
    x0[idx_v] = v_0[pq_idx]
    x = x0.copy()

    def pf_equations(x):
        v_ph = x_to_v(x)
        S_calc = v_ph * np.conj(y_bus.dot(v_ph))
        # S_err = p_sum_bus + 1j*q_sum_bus + S_calc
        p_err = p_sum_bus + S_calc.real
        q_err = q_sum_bus + S_calc.imag

        return np.concatenate([p_err[pvpq_idx], q_err[pq_idx]])

    converged = False
    i = 0
    x = x0.copy()
    err = pf_equations(x)
    err_norm = max(abs(err))

    while not converged and i < pf_max_it:
        i = i + 1

        # Numerical jacobian
        J = jacobian_num(pf_equations, x)

        # Update step
        dx = np.linalg.solve(J, err)
        x -= dx

        err = pf_equations(x)
        err_norm = max(abs(err))

        if tol > err_norm:
            converged = True
            # if print_output:
            #     print('Power flow converged.')
        if i == pf_max_it:
            print('Warning: Power flow did not converge in {} iterations.'.format(pf_max_it))

    v_sol = x_to_v(x)
    s_sol = v_sol * np.conj(y_bus.dot(v_sol))

    return v_sol, s_sol, converged


def remove_recarray_field(a, field):
    names = []
    col_dtypes = []
    data = []
    for i, name in enumerate(a.dtype.names):
        if not name == field:
            # print(i, name)
            names.append(name)
            col_dtypes.append(a.dtype[i])
            data.append(a[name])

    dtypes = [(name_, dtype_) for name_, dtype_ in zip(names, col_dtypes)]

    entries = [entry for entry in zip(*data)]
    return np.array(entries, dtype=dtypes)


def combine_recarrays(a, b):
    '''
    Combine columns of recarrays
    :param a:
    :param b:
    :return:
    '''
    new_dtype = a.dtype.descr + b.dtype.descr
    c = np.zeros(a.shape, new_dtype)
    for name in a.dtype.names:
        c[name] = a[name]

    for name in b.dtype.names:
        c[name] = b[name]

    return c


def replace_str_col(a, col, strings):
    '''
    Replaces column with strings (col) in structured array (a) with new strings.
    dtype is updated if new strings are longer than previous strings.
    :param a: Structured array
    :param col: String that points to column of a
    :param strings: Array of strings that will replace content in a[col]
    :return: Structured array with updated column
    '''
    new_strings = np.array(strings, dtype=[(col, strings.dtype)])
    a_old = remove_recarray_field(a, col)
    return combine_recarrays(new_strings, a_old)


def concatenate_structured_arrays(a_list):
    '''
    Combine rows of recarrays
    :param a_list:
    :return:
    '''
    entries = []
    for a in a_list:
        for row in a:
            entries.append(row)

    header = a_list[0].dtype.names
    # # Get dtype for each column
    # # data = model[td][1:]
    # entries_T = list(map(list, zip(*entries)))
    # col_dtypes = [np.array(col).dtype for col in entries_T]
    #
    # # entries = [tuple(entry) for entry in data]
    # dtypes = [(name_, dtype_) for name_, dtype_ in zip(header, col_dtypes)]

    # return np.array(entries, dtype=dtypes)
    return structured_array_from_list(header, entries)


def structured_array_from_list(names, entries):
    entries_T = list(map(list, zip(*entries)))
    col_dtypes = [np.array(col).dtype for col in entries_T]

    entries_tup = [tuple(entry) for entry in entries]
    dtypes = [(name_, dtype_) for name_, dtype_ in zip(names, col_dtypes)]

    return np.array(entries_tup, dtype=dtypes)


def lookup_strings(a, b, return_mask=False):
    # Function to find the index of the element in b that equal the element in a, for each element in a
    if isinstance(a, np.ndarray) or isinstance(a, list):
        lookups = []
        found = []
        for a_ in a:
            lookup = np.where(b == a_)[0]
            if len(lookup) > 0:
                lookups.append(lookup[0])
                found.append(True)
            else:
                found.append(False)
        if return_mask:
            return np.array(lookups), np.array(found)
        else:
            return np.array(lookups)
    else:
        lookup = np.where(b == a)[0]
        if len(lookup) > 0:
            return lookup[0]
        else:
            return np.nan


def jacobian_num(f, x, eps=1e-10, **params):
    # Numerical computation of Jacobian
    J = np.zeros([len(x), len(x)], dtype=float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        f1 = f(x1, **params)
        f2 = f(x2, **params)

        J[:, i] = (f1 - f2) / (2 * eps)

    return J


class DynamicModel:  # This is not used anymore?
    # Empty dummy-class for dynamic models (Gen, AVR, GOV, PSS etc.)
    def __init__(self):
        pass

class EventManager:
    # Not in use?
    def __init__(self, events, event_function):
        self.events = events
        self.event_flags = np.ones(len(self.events), dtype=bool)
        self.event_function = event_function

    def update(self, t_now):
        for i, (t_event, sub_events) in enumerate(self.events):
            if t_now >= t_event and self.event_flags[i]:
                self.event_flags[i] = False
                for element_type, name, action in sub_events:
                    self.event_function(element_type, name, action)
                    print(name + ' was ' + action + 'ed.')


def newton_raphson_power_flow_distributed_slack(
        y_bus, v_0, p_sum_bus, q_sum_bus, bus_types,
        participation, angle_ref_idx, tol, pf_max_it):
    """Solve a power flow with active-power mismatch participation."""
    n_bus = len(bus_types)
    participation = np.asarray(participation, dtype=float)
    if participation.shape != (n_bus,):
        raise ValueError('participation must have length n_bus.')
    if not np.isclose(np.sum(participation), 1.0):
        raise ValueError('participation factors must sum to 1.')
    if not 0 <= angle_ref_idx < n_bus:
        raise ValueError('angle_ref_idx must be a valid bus index.')

    pv_idx = np.where(bus_types == 'PV')[0]
    pq_idx = np.where(bus_types == 'PQ')[0]
    variable_angle_idx = np.array([idx for idx in range(n_bus) if idx != angle_ref_idx])
    pvpq_idx = np.concatenate([pv_idx, pq_idx])
    if not np.array_equal(np.sort(pvpq_idx), np.arange(n_bus)):
        raise ValueError("bus_types must contain only 'PV' and 'PQ'.")

    n_angle = len(variable_angle_idx)
    n_voltage = len(pq_idx)
    d_p_idx = n_angle + n_voltage

    def x_to_v(x):
        phi = np.zeros(n_bus)
        phi[variable_angle_idx] = x[:n_angle]
        voltage = np.asarray(v_0, dtype=float).copy()
        voltage[pq_idx] = x[n_angle:n_angle + n_voltage]
        return voltage * np.exp(1j * phi)

    p_input = np.asarray(p_sum_bus, dtype=float).copy()
    q_input = np.asarray(q_sum_bus, dtype=float).copy()

    def pf_equations(x):
        voltage = x_to_v(x)
        s_calc = voltage * np.conj(y_bus.dot(voltage))
        d_p = x[d_p_idx]
        p_err = p_input + participation * d_p + s_calc.real
        q_err = q_input + s_calc.imag
        return np.concatenate([p_err, q_err[pq_idx]])

    x = np.zeros(d_p_idx + 1)
    x[:n_angle] = 0.0
    x[n_angle:n_angle + n_voltage] = np.asarray(v_0, dtype=float)[pq_idx]

    converged = False
    for iteration in range(pf_max_it):
        err = pf_equations(x)
        if np.max(np.abs(err)) < tol:
            converged = True
            break
        jacobian = jacobian_num(pf_equations, x)
        x -= np.linalg.solve(jacobian, err)
    else:
        print('Warning: Distributed-slack power flow did not converge in {} iterations.'.format(pf_max_it))

    v_sol = x_to_v(x)
    s_sol = v_sol * np.conj(y_bus.dot(v_sol))
    return v_sol, s_sol, x[d_p_idx], converged


