import numpy as np
import sympy as sym
import networkx as nx

import scipy.integrate


# Base cohesin reaction network object
class CohesinNetwork(nx.MultiDiGraph):

    def __init__(self, transitions):

        super().__init__(transitions)

        self.N_transitions = self.number_of_edges()
        self.accessory_proteins = [s[-1] for s in self.nodes if s not in ['R', 'RB']]

        self.unbound_states = ['R'] + self.accessory_proteins
        self.bound_states = [s for s in self.nodes if s != 'R']

        self.full_states = self.bound_states + self.unbound_states
        self.state_ids = {k: v for v, k in enumerate(self.full_states)}

        assert len(self.full_states) == self.N_transitions

        self.make_symbols()

        self.make_rhs()
        self.make_rhs_eq()


    def make_symbols(self):

        self.states = {k: sym.symbols(k) for k in self.full_states}
        self.rates = {(s1, s2): sym.symbols(f"k_{s1[-1]}{s2[-1]}") for s1, s2 in self.edges()}

        self.abundances = {k: sym.symbols(f"N_{k}") for k in self.unbound_states}
        self.bound_fractions = {k: sym.symbols(f"F_{k}") for k in self.unbound_states}
        self.residence_times = {k: sym.symbols(f"tau_{k}") for k in self.unbound_states}


    def make_rhs(self):

        rhs = sym.zeros(self.N_transitions, 1)

        for s1, s2 in self.edges():
            if s2 not in ['R', 'RB']:
                rhs[self.state_ids[s1]] -= self.rates[(s1, s2)] * self.states[s1] * self.states[s2[-1]]
                rhs[self.state_ids[s2]] += self.rates[(s1, s2)] * self.states[s1] * self.states[s2[-1]]

            else:
                rhs[self.state_ids[s1]] -= self.rates[(s1, s2)] * self.states[s1]
                rhs[self.state_ids[s2]] += self.rates[(s1, s2)] * self.states[s1]

        for p in self.accessory_proteins:
            rhs[self.state_ids[p]] = -rhs[self.state_ids[f"R{p}"]]

        self.rhs = rhs


    def make_rhs_eq(self):

        rhs_eq = sym.zeros(self.N_transitions, 1)

        ## Set substitution dictionaries for equilibrium concentrations
        subs_dict = {'RB': self.abundances['R'] * self.bound_fractions['R'],
                     'R': self.abundances['R'] * (1 - self.bound_fractions['R'])}
        
        for p in self.accessory_proteins:
            subs_dict[f"R{p}"] = self.abundances[p] * self.bound_fractions[p]
            subs_dict[p] = self.abundances[p] * (1 - self.bound_fractions[p])

            subs_dict['RB'] -= subs_dict[f"R{p}"]

        subs_R = {k: v for k, v in subs_dict.items() if k not in ['R']}
        eq_R = self.rhs[self.state_ids['R']].subs(subs_R)

        # Equilibrium RAD21 bound fractions & residence times
        rhs_bound_R = 1 / self.residence_times['R'] * self.bound_fractions['R'] / (1-self.bound_fractions['R'])
        rhs_residence_R = -1 / self.residence_times['R'] * self.bound_fractions['R']*self.abundances['R']

        rhs_eq[self.state_ids['R']] = sym.collect(eq_R, self.states['R']).coeff(self.states['R'], 1) + rhs_bound_R
        rhs_eq[self.state_ids['RB']] = sym.collect(eq_R, self.states['R']).coeff(self.states['R'], 0) + rhs_residence_R

        for p in self.accessory_proteins:
            subs = {k: v for k, v in subs_dict.items() if k not in [p, f"R{p}"]}
            eq = self.rhs[self.state_ids[p]].subs(subs)

            # Equilibrium accessory protein bound fractions & residence times
            rhs_bound = 1 / self.residence_times[p] * self.bound_fractions[p] / (1-self.bound_fractions[p])
            rhs_residence = -1 / self.residence_times[p]

            rhs_eq[self.state_ids[p]] = sym.collect(eq, self.states[p]).coeff(self.states[p], 1) + rhs_bound
            rhs_eq[self.state_ids[f"R{p}"]] = sym.collect(eq, self.states[f"R{p}"]).coeff(self.states[f"R{p}"], 1) + rhs_residence

        self.rhs_eq = rhs_eq


    def solve_rates(self, parameter_dict_wt):

        rate_dict = {}
        sol_rates = sym.solve(self.rhs_eq, *self.rates.values())

        for rate, expr in sol_rates.items():
            value = expr.evalf(subs=parameter_dict_wt)
            rate_dict[str(rate)] = value

        self.rate_dict = rate_dict
        self.parameter_dict_wt = parameter_dict_wt


    def solve_kinetics(self, parameter_dict, t_max=3600, n_steps=10000):

        if hasattr(self, 'rate_dict'):
            # Compute model kinetics, starting from fully unbound population of RAD21
            t = sym.symbols('t')
            init_state = sym.Matrix([[0]*len(self.bound_states) + list(self.abundances.values())])

            t_span = (0, t_max)
            t_eval = np.linspace(0, t_max, n_steps)

            y0_eval = list(init_state.evalf(subs=parameter_dict))
            rhs_eval = list(self.rhs.evalf(subs=self.rate_dict))

            f = sym.lambdify((t, self.states.values()), rhs_eval)
            solution = scipy.integrate.solve_ivp(f, t_span, y0_eval, t_eval=t_eval)

            return solution
        
        else:
            print("Network lacks fitted rate values - please run 'solve_rates' method first")


    def to_extrusion_dict(self, parameter_dict,
	                      reference_dict={"LEF_transition_rates": {}},
	                      t=10000,
	                      site_types=['A'],
	                      fix_velocity=False,
	                      active_states=['RN']):
		
        output_dict = reference_dict.copy()
        output_dict["LEF_states"] = {k: v+1 for v, k in enumerate(self.sequence)}

        kinetics = self.solve_kinetics(parameter_dict, t_max=t)
        final_state = kinetics.y[:,-1]

        for rate, value in self.rate_dict.items():
            s1 = f"R{rate[-2]}"
            s2 = f"R{rate[-1]}"

            id1 = self.sequence.index(s1)+1 if s1 in self.sequence else 0
            id2 = self.sequence.index(s2)+1 if s2 in self.sequence else 0

            rate_key = f"{id1}{id2}"
		
            if rate[-1] not in ['R', 'B']:
                k = self.full_states.index(rate[-1])
                value *= final_state[k]

            if (id1 != 0) & (id2 != 0):
                output_dict["LEF_transition_rates"][rate_key] = {t: float(value) for t in site_types}
            elif id1 == 0:
                output_dict["LEF_on_rate"] = {t: float(value) for t in site_types}
            elif id2 == 0:
                output_dict["LEF_off_rate"] = {t: float(value) for t in site_types}
                output_dict["LEF_stalled_off_rate"] = {t: float(value) for t in site_types}

        if fix_velocity:
            if "velocity_multiplier" in reference_dict.keys():
                active_indices = [self.full_states.index(state) for state in active_states]
                active_levels = [final_state[index] for index in active_indices]

                rescale_velocity = parameter_dict['F_R'] * parameter_dict['N_R'] / sum(active_levels)

                output_dict["velocity_multiplier"] *= rescale_velocity

            else:
                print("fix_velocity option requires velocity_multiplier to be defined in the reference extrusion dictionary")

        return output_dict
