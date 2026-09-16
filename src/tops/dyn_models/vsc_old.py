from tops.dyn_models.blocks import *
from .pll import PLL1

class VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.pi_p = PIRegulator(K_p=p['P_K_p'], K_i=p['P_K_i'])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p['Q_K_p'], K_i=p['Q_K_i'])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p['T_i'])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_q(x, v)

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))

        self.pi_p.initialize(
            x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0)
        )

        self.pi_q.initialize(
            x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0)
        )

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]


class VSC_SI(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=p['T_pll'], bus=p['bus'])
        self.pll.sys_par = self.sys_par

        self.p_ref_filter = TimeConstant(T=p['T_p'])
        self.p_ref_filter.input = lambda x, v: self.p_ref(x, v)*p['S_n']

        self.q_ref_filter = TimeConstant(T=p['T_q'])
        self.q_ref_filter.input = lambda x, v: self.q_ref(x, v)*p['S_n']

        self.si_filter = TimeConstant(T=p['T_SI'])
        self.si_filter.input = lambda x, v: self.pll.freq_est(x, v)

        self.current_d = TimeConstant(T=p['T_i'])
        self.current_d.input = self._current_d_command
        self.current_q = TimeConstant(T=p['T_i'])
        self.current_q.input = self._current_q_command

    def input_list(self):
        return ['p_ref', 'q_ref']

    def _voltage_denominator(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        return np.maximum(np.sqrt(3)*abs(v[self.bus_idx_red['terminal']])*v_n, 1e-9)

    def _synthetic_power(self, x, v):
        p_si = self.par['K_SI']*self.si_filter.output(x, v)*self.par['S_n']
        p_si_max = self.par['P_SI_max']*self.par['S_n']
        return np.minimum(np.maximum(p_si, -p_si_max), p_si_max)

    def _active_power_reference(self, x, v):
        return self.p_ref_filter.output(x, v) + self._synthetic_power(x, v)

    def _reactive_power_reference(self, x, v):
        return self.q_ref_filter.output(x, v)

    def _current_d_command(self, x, v):
        denominator = self._voltage_denominator(x, v)
        reference = self._active_power_reference(x, v)
        return (reference + self.par['k_p']*(reference - self.P(x, v)))/denominator

    def _current_q_command(self, x, v):
        denominator = self._voltage_denominator(x, v)
        reference = self._reactive_power_reference(x, v)
        return (reference + self.par['k_q']*(reference - self.Q(x, v)))/denominator

    def _limited_currents(self, x, v):
        current_d = self.current_d.output(x, v)
        current_q = self.current_q.output(x, v)
        current_abs = np.sqrt(current_d**2 + current_q**2)
        scale = np.minimum(1, self.par['i_max']/np.maximum(current_abs, 1e-9))
        return current_d*scale, current_q*scale

    def I_inj(self, x, v):
        current_d, current_q = self._limited_currents(x, v)
        return (current_d - 1j*current_q)*np.exp(1j*self.pll.output(x, v))

    def P(self, x, v):
        current_d, _ = self._limited_currents(x, v)
        return self._voltage_denominator(x, v)*current_d

    def Q(self, x, v):
        _, current_q = self._limited_currents(x, v)
        return self._voltage_denominator(x, v)*current_q

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], -self.par['q_ref']*self.par['S_n']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['p_ref'] = self.par['p_ref']
        self._input_values['q_ref'] = self.par['q_ref']

        p_ref = self.par['p_ref']*self.par['S_n']
        q_ref = self.par['q_ref']*self.par['S_n']
        denominator = np.maximum(
            np.sqrt(3)*abs(v_0[self.bus_idx_red['terminal']])*self.sys_par['bus_v_n'][self.bus_idx_red['terminal']],
            1e-9
        )

        self.p_ref_filter.initialize(x_0, v_0, p_ref)
        self.q_ref_filter.initialize(x_0, v_0, q_ref)
        self.si_filter.initialize(x_0, v_0, np.zeros(self.n_units))
        self.current_d.initialize(x_0, v_0, p_ref/denominator)
        self.current_q.initialize(x_0, v_0, q_ref/denominator)

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]