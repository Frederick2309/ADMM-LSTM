""" admm.py (c) Yang Yang, 2024-2025
This file implements an ADMM-based optimizer which is designed to train an LSTM-Linear model defined in blocks/lstm.py.
"""

import torch
from blocks.lstm import LSTM
from typing import Dict, Tuple, Callable
from _global import info, warning, log_assert, error, device
from parameters import example_parameter_dictionary


with_dual_y = False


def compute_grad(func, x0):
    x0 = x0.clone().detach().to(torch.float).requires_grad_(True)
    func_result = func(x0)
    func_result.backward()
    return x0.grad


class ADMMBasedOptimizer(object):
    """
    This class implements an ADMM-based optimizer designed to optimize an LSTM-Linear model defined in blocks/lstm.py.

    The one-step update process follows the following procedure:
      1. Update Wy, Wi, Vi, Wf, Vf, Wg, Vg, Wo, Vo
      2. For t = 1, 2, ..., seq_len:
        - Update i, f, g, o, c, h (at t)
        - Update dual variables (at t)

    """

    def __init__(self, model: LSTM,
                 training_samples: Tuple[torch.Tensor, torch.Tensor],
                 parameter_dictionary: Dict[str, Dict[str, float]] = None,
                 verbose: bool = True) -> None:
        """
        Initialize the ADMM-based optimizer class.
        :param model: (LSTM) An object initialized with LSTM defined in blocks/lstm.py
        :param training_samples: ((train_x, train_y)) A tuple containing the training samples.
        :param parameter_dictionary: ({'rho': {...}, 'beta': {...}}) A dictionary defining the parameters (See example).
        :param verbose: (bool) If true, a summary will be printed at each key stage of the training.
        """
        self.model = model.to(device)
        self.train_x, self.train_y = training_samples
        self.batch_size, self.seq_len, self.input_size, self.output_size, self.hidden_size = \
            self.__initialize_training_constants(training_samples)
        self.verbose = verbose

        self.betas, self.rhos = dict(), dict()
        self.__set_summary(clear=True)
        self.__initialize_normalization_factors(parameter_dictionary)
        self.__initialize_penalties(parameter_dictionary)

        self.gates = dict()
        self.__initialize_primal_gates()

        self.duals = dict()
        self.__initialize_dual_variables()

    def step(self) -> None:
        """
        Implement the one-step update process.
        :return: (None).
        """
        # with torch.no_grad():
        self.__update_wy()
        for map_to in ['i', 'f', 'g', 'o']:
            for map_from in ['x', 'h']:
                self.__update_weights(map_from, map_to)
        for t in range(1, self.seq_len + 1):
            self.__update_gates(t)
            if t == self.seq_len:
                self.__update_primal_a()
            self.__update_duals(t)
        if with_dual_y:
            self.__update_dual_y()

    def __set_summary(self, clear: bool = False, display: bool = False, msg: str = None) -> None:
        try:
            summary = getattr(self, 'summary')
            if summary and display:
                info(summary)
            if clear:
                setattr(self, 'summary', None)
            if msg:
                setattr(self, 'summary', summary if summary else str() + msg)
        except AttributeError:
            setattr(self, 'summary', None)

    def __initialize_training_constants(self, training_samples: Tuple[torch.Tensor, torch.Tensor]
                                        ) -> Tuple[int, int, int, int, int]:
        train_x, train_y = training_samples

        train_batch, train_seq, train_feat = train_x.size()
        train_label_batch, train_label_feat = train_y.size()

        log_assert(train_batch == train_label_batch,
                   f'Batch size of samples mismatch '
                   f'(Got train_x: {train_batch}, train_y: {train_label_batch}).')

        log_assert(train_feat == self.model.input_size and train_label_feat == self.model.output_size,
                   f'Input and output size of samples must match that of the model '
                   f'(Got train_x: {train_feat}, train_y: {train_label_feat}, '
                   f'model: {self.model.input_size} -> {self.model.output_size}).')

        return train_batch, train_seq, train_feat, train_label_feat, self.model.hidden_size

    def __initialize_normalization_factors(self, param_dict: Dict[str, Dict[str, float]]) -> None:
        if not param_dict:
            example_dict = example_parameter_dictionary['GoogleStock']
            warning(f'Parameter dictionary is empty, a default one will be applied: '
                    f'{{\n    \'beta\': {example_dict["beta"]},\n'
                    f'    \'rho\': {example_dict["rho"]}\n}})')
            self.betas, self.rhos = [{
                key: torch.tensor(value) for key, value in example_dict[dict_key].items()
            } for dict_key in ['beta', 'rho']]
            return
        try:
            beta_dict = param_dict['beta']
        except KeyError:
            error('Normalization factors missing in parameter dictionary.')
        self.__set_summary(False, False, 'Parameters:\n  {\n    \'beta\': {')

        def assertion(key):
            value = beta_dict[key]
            log_assert(isinstance(value, float) or isinstance(value, int), f'Beta {key} must be a float or integer.')
            log_assert(beta_dict[key] >= 0, f'Beta {key} must be non-negative.')

        try:
            assertion('wy')
            self.betas['wy'] = torch.tensor(beta_dict['wy'], dtype=torch.float, device=device)
        except KeyError:
            error('Key wy missing in normalization factors.')
        for weight_type in ['w', 'v']:
            for map_to in ['i', 'f', 'g', 'o']:
                key = weight_type + map_to
                try:
                    assertion(key)
                    self.betas[f'{"x" if weight_type == "w" else "h"}2{map_to}'] = (
                        torch.tensor(beta_dict[key], dtype=torch.float, device=device))
                    self.__set_summary(False, False, f'\'{weight_type}{map_to}\': {beta_dict[key]: .6f}'
                                                     + '}\n' if weight_type == 'v' and map_to == 'o' else ', ')
                except KeyError:
                    error(f'Key {key} missing in normalization factors.')

    def __initialize_penalties(self, param_dict: Dict[str, Dict[str, float]]) -> None:
        if not param_dict:
            return
        log_assert('rho' in param_dict.keys(), 'Penalties missing in parameter dictionary.')
        rho_dict = param_dict['rho']
        self.__set_summary(False, False, '    \'rho\': {')
        for gate in ['i', 'f', 'g', 'o', 'c', 'h', 'y']:
            log_assert(gate in rho_dict.keys(), 'Penalties missing in parameter dictionary.')
            log_assert(isinstance(rho_dict[gate], float) or isinstance(rho_dict[gate], int),
                       f'Penalties must be a float or integer.')
            self.rhos[gate] = torch.tensor(rho_dict[gate], dtype=torch.float, device=device)
            self.__set_summary(False, False, f'\'{gate}\': {rho_dict[gate]: .6f}'
                                             + '}\n' if gate == 'y' else ', ')
        if self.verbose:
            self.__set_summary(True, False)

    def __initialize_primal_gates(self) -> None:
        initial_prediction = self.model.init_gate_variables(self.train_x, None, None)
        for key, value in initial_prediction.items():
            self.gates[key] = value

    def __initialize_dual_variables(self) -> None:
        for gate in ['i', 'f', 'g', 'o', 'c', 'h']:
            self.duals[gate] = torch.zeros((self.batch_size, self.seq_len + 1, self.hidden_size),
                                           dtype=torch.float, device=device)
        self.duals['y'] = torch.zeros((self.batch_size, self.output_size), dtype=torch.float, device=device)

    def __get_weight(self, map_from: str, map_to: str) -> torch.Tensor:
        return self.model.get_weight(map_from, map_to)

    def __set_weight(self, map_from: str, map_to: str, value: torch.Tensor) -> None:
        self.model.set_weight(map_from, map_to, value)

    def __get_wy(self) -> torch.Tensor:
        return self.model.get_wy()

    def __set_wy(self, value: torch.Tensor) -> None:
        self.model.set_wy(value)

    def __get_gate(self, gate: str, t: int = None) -> torch.Tensor:
        if t is not None:
            if t == -1:
                t = self.seq_len
            return self.gates[gate][:, t, :].clone().detach()
        return self.gates[gate].clone().detach()

    def __set_gate(self, gate: str, t: int, value: torch.Tensor = None) -> None:
        self.gates[gate][:, t, :] = value.clone().detach()

    def __get_a(self) -> torch.Tensor:
        return self.gates['a'].clone().detach()

    def __set_a(self, value: torch.Tensor) -> None:
        self.gates['a'] = value.clone().detach()

    def __get_dual(self, gate: str, t: int) -> torch.Tensor:
        return self.duals[gate][:, t, :].clone().detach()

    def __set_dual(self, gate: str, t: int, value: torch.Tensor) -> None:
        self.duals[gate][:, t, :] = value.clone().detach()

    def __get_dual_y(self) -> torch.Tensor:
        return self.duals['y'].clone().detach()

    def __set_dual_y(self, value: torch.Tensor) -> None:
        self.duals['y'] = value.clone().detach()

    def __get_rho(self, gate: str) -> torch.Tensor:
        return self.rhos[gate].clone().detach()

    def __get_beta(self, map_from: str, map_to: str) -> torch.Tensor:
        return self.betas[f'{map_from}2{map_to}'].clone().detach()

    def __get_beta_y(self) -> torch.Tensor:
        return self.betas['wy'].clone().detach()

    @staticmethod
    def __inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y)

    def __fro_quad(self, x: torch.Tensor) -> torch.Tensor:
        return self.__inner_product(x, x)

    @staticmethod
    def __tanh(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def __sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def __d_tanh(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - self.__tanh(x) ** 2

    def __d_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        sig = self.__sigmoid(x)
        return sig * (1 - sig)

    def __update_wy(self) -> None:
        h = self.__get_gate('h', self.seq_len)
        a = self.__get_a()
        wy = self.__get_wy()
        rho_y = self.__get_rho('y')
        theta = 1

        def original_func(beta_t: torch.Tensor) -> torch.Tensor:
            return 0.5 * rho_y * self.__fro_quad(h @ beta_t - a - (
                0 if not with_dual_y else (
                    self.__get_dual_y() / rho_y
                )
            ))

        gradient = compute_grad(original_func, wy)

        def estimated_func(beta_t: torch.Tensor, theta_t: float) -> torch.Tensor:
            return (original_func(beta_t) + self.__inner_product(gradient, beta_t - wy)
                    + 0.5 * theta_t * self.__fro_quad(beta_t - wy))

        def compute_wy(theta):
            return (theta * wy - gradient) / (theta + self.__get_beta_y())

        beta = wy + gradient / theta
        # beta = compute_wy(theta)

        while original_func(beta) > estimated_func(beta, theta):
            theta *= 2
            beta = wy + gradient / theta
            # beta = compute_wy(theta)

        theta /= 2

        # self.__set_wy((theta * wy - gradient) / (theta + 2 * self.__get_beta_y()))
        self.__set_wy(compute_wy(theta))

    def __update_weights(self, map_from: str, map_to: str) -> None:
        w = self.__get_weight(map_from, map_to)
        rho_gate = self.__get_rho(map_to)

        if map_to == 'g':
            act_fun: Callable[[torch.Tensor], torch.Tensor] = self.__tanh
            d_act_fun: Callable[[torch.Tensor], torch.Tensor] = self.__d_tanh
        else:
            act_fun: Callable[[torch.Tensor], torch.Tensor] = self.__sigmoid
            d_act_fun: Callable[[torch.Tensor], torch.Tensor] = self.__d_sigmoid

        if map_from == 'x':
            map_from_value = self.train_x
            the_other_map_from_value = self.__get_gate('h')
            the_other_weight = self.__get_weight('h', map_to)
        else:
            map_from_value = self.__get_gate('h')
            the_other_map_from_value = self.train_x
            the_other_weight = self.__get_weight('x', map_to)

        def compute_grad() -> torch.Tensor:
            _gradient = torch.zeros_like(w, dtype=torch.float, device=device)
            for t in range(1, self.seq_len + 1):
                map_from_value_t = map_from_value[:, t - 1, :]
                the_other_map_from_value_t = the_other_map_from_value[:, t - 1, :]
                act_value = map_from_value_t @ w + the_other_map_from_value_t @ the_other_weight
                _gradient += (
                    map_from_value_t.T @ ((act_fun(act_value) - self.__get_dual(map_to, t) / rho_gate
                                           - self.__get_gate(map_to, t)) * d_act_fun(act_value))
                )
            return _gradient * rho_gate

        gradient = compute_grad()

        def original_func(beta_t: torch.Tensor) -> torch.Tensor:
            func_value = torch.tensor(0., dtype=torch.float, device=device)
            for t in range(1, self.seq_len + 1):
                map_from_value_t = map_from_value[:, t - 1, :]
                the_other_map_from_value_t = the_other_map_from_value[:, t - 1, :]
                func_value += 0.5 * self.__get_rho(map_to) * self.__fro_quad(
                    act_fun(map_from_value_t @ beta_t + the_other_map_from_value_t @ the_other_weight)
                    - self.__get_dual(map_to, t) / rho_gate - self.__get_gate(map_to, t)
                )
            return func_value

        def estimated_func(beta_t: torch.Tensor, theta_t: float) -> torch.Tensor:
            return (original_func(w) + torch.sum(gradient * (beta_t - w))
                    + self.seq_len * 0.5 * theta_t * self.__fro_quad(beta_t - w))

        theta = 1
        beta = w + gradient / theta

        while original_func(beta) > estimated_func(beta, theta):
            theta *= 2
            beta = w + gradient / theta

        theta /= 2

        self.__set_weight(map_from, map_to, (
            (0.5 * rho_gate * self.seq_len * theta * w - gradient)
            / (self.__get_beta(map_from, map_to) + 0.5 * rho_gate * theta * self.seq_len)
        ))

    def __update_gates(self, t: int) -> None:
        self.__update_primal_i_f_g_o('i', t)
        self.__update_primal_i_f_g_o('f', t)
        self.__update_primal_i_f_g_o('g', t)
        self.__update_primal_i_f_g_o('o', t)
        self.__update_primal_c(t)
        self.__update_primal_h(t)

    def __update_primal_i_f_g_o(self, gate: str, t: int) -> None:
        act_value = (self.train_x[:, t - 1, :] @ self.__get_weight('x', gate)
                     + self.__get_gate('h', t - 1) @ self.__get_weight('h', gate))
        act_func: Callable[[torch.Tensor], torch.Tensor] = self.__tanh if gate == 'g' else self.__sigmoid
        p1, p2, p3, rho_var2, lam_var2, var2 = None, None, None, None, None, None
        rho_var1 = self.__get_rho(gate)
        lam_var = self.__get_dual(gate, t)
        match gate:
            case 'i':
                p1 = self.__get_gate('g', t)
                p2 = self.__get_gate('f', t)
                p3 = self.__get_gate('c', t - 1)
            case 'f':
                p1 = self.__get_gate('c', t - 1)
                p2 = self.__get_gate('g', t)
                p3 = self.__get_gate('i', t)
            case 'g':
                p1 = self.__get_gate('i', t)
                p2 = self.__get_gate('f', t)
                p3 = self.__get_gate('c', t - 1)
            case 'o':
                p1 = self.__tanh(self.__get_gate('c', t))
                p2, p3 = 0., 0.
        if gate == 'o':
            var2 = self.__get_gate('h', t)
            rho_var2 = self.__get_rho('h')
            lam_var2 = self.__get_dual('h', t)
        else:
            var2 = self.__get_gate('c', t)
            rho_var2 = self.__get_rho('c')
            lam_var2 = self.__get_dual('c', t)
        self.__set_gate(gate, t, - (
            lam_var - rho_var1 * act_func(act_value) + (rho_var2 * (p2 * p3 - var2) - lam_var2) * p1
        ) / (rho_var1 + rho_var2 * p1 * p1))

    def __update_primal_c(self, t: int) -> None:
        c = self.__get_gate('c', t)
        o = self.__get_gate('o', t)
        h = self.__get_gate('h', t)
        div_h = self.__get_dual('h', t) / self.__get_rho('h')
        div_c = self.__get_dual('c', t) / self.__get_rho('c')
        rho_h = self.__get_rho('h')
        rho_c = self.__get_rho('c')

        # non-iterative
        # gradient = rho_h * self.__d_tanh(c) * o * (self.__tanh(c) * o - h - div_h)
        # theta = torch.max(torch.abs((h - div_h)))
        # self.__set_gate('c', t, - (self.__get_dual('c', t) - rho_c * (
        #     self.__get_gate('f', t) * self.__get_gate('c', t - 1)
        #     + self.__get_gate('i', t) * self.__get_gate('g', t)
        # ) + rho_h * (gradient - 0.5 * theta * c)) / (rho_c + 0.5 * theta * rho_h))

        # use iterative method:
        assistant_o = o
        assistant_z = h + div_h

        def original_func(c_t):
            return .5 * self.__fro_quad(
                self.__tanh(c_t) * assistant_o - assistant_z
            )

        gradient = compute_grad(original_func, c)
        current_original_func = original_func(c)

        def assistant_func(c_t, theta_t):
            return current_original_func + self.__inner_product(gradient, c_t - c) + .5 * theta_t * self.__fro_quad(
                c_t - c
            )

        A = (div_c - self.__get_gate('f', t) * self.__get_gate('c', t - 1)
             - self.__get_gate('i', t) * self.__get_gate('g', t))

        def compute_c(theta_t):
            return (theta_t * c - gradient - rho_c * A) / (rho_c + theta_t)

        theta = 1
        current_c = c.detach().clone()
        while original_func(current_c) > assistant_func(current_c, theta):
            theta *= 2
            current_c = compute_c(theta)

        theta /= 2

        self.__set_gate('c', t, compute_c(theta))


    def __update_primal_h(self, t: int) -> None:
        h = self.__get_gate('h', t)
        wy = self.__get_wy()
        rho_h = self.__get_rho('h')
        lam_h = self.__get_dual('h', t)
        a = self.__get_a()
        o = self.__get_gate('o', t)
        tanh_c = self.__tanh(self.__get_gate('c', t))
        theta = 0.1
        # theta = 0.5
        theta_max = 1
        # theta_max = 100
        # gradient = rho_h * ((h @ wy - a - (
        #     0 if not with_dual_y else (self.__get_dual_y() / self.__get_rho('y'))
        # )) @ wy.T)

        if t < self.seq_len:
            self.__set_gate('h', t, (rho_h * o * tanh_c - lam_h) / rho_h)
            return

        def original_func(beta_t: torch.Tensor) -> torch.Tensor:
            return 0.5 * self.__get_rho('y') * self.__fro_quad(beta_t @ wy - a - (
                0 if not with_dual_y else (self.__get_dual_y() / self.__get_rho('y'))
            ))

        gradient = compute_grad(original_func, h)

        def estimated_func(beta_t: torch.Tensor, theta_t: float) -> torch.Tensor:
            return (original_func(h) + torch.sum(gradient * (beta_t - h))
                    + 0.5 * theta_t * self.__fro_quad(beta_t - h))

        def compute_h(theta):
            return (theta * h + rho_h * o * tanh_c - lam_h - gradient) / (theta + rho_h)

        # beta = gradient / theta
        beta = compute_h(theta)
        while original_func(beta) > estimated_func(beta, theta):
            theta *= 2
            # beta = gradient / theta
            beta = compute_h(theta)
            if theta >= theta_max:
                break

        theta /= 2
        self.__set_gate('h', t, (
            (theta * h + rho_h * o * tanh_c
             - lam_h - gradient) / (theta + rho_h)
            # compute_h(theta)
        ))

    def __update_primal_a(self) -> None:
        rho_y = self.__get_rho('y')
        # self.__set_a(
        #     (self.batch_size * rho_y * (self.__get_gate('h', self.seq_len) @ self.__get_wy()) + 2 * self.train_y)
        #     / (2 + self.batch_size * rho_y)
        # )

        self.__set_a(
            (2 * self.train_y + self.batch_size * rho_y * (self.__get_gate('h', self.seq_len) @ self.__get_wy())
             - (0 if not with_dual_y else (
                            self.batch_size * self.__get_dual_y()
                    )))
            / (2 + self.batch_size * rho_y)
        )

    def __update_duals(self, t: int) -> None:
        self.__update_dual_i_f_g_o('i', t)
        self.__update_dual_i_f_g_o('f', t)
        self.__update_dual_i_f_g_o('g', t)
        self.__update_dual_i_f_g_o('o', t)
        self.__update_dual_c(t)
        self.__update_dual_h(t)

    def __update_dual_i_f_g_o(self, gate: str, t: int) -> None:
        x_t = self.train_x[:, t - 1, :]
        h_before = self.__get_gate('h', t - 1)
        act_func: Callable[[torch.Tensor], torch.Tensor] = self.__tanh if gate == 'g' else self.__sigmoid
        self.__set_dual(gate, t, (
            self.__get_dual(gate, t) + self.__get_rho(gate) * (
                self.__get_gate(gate, t) - act_func(
                    x_t @ self.__get_weight('x', gate) + h_before @ self.__get_weight('h', gate)
                )
            )
        ))

    def __update_dual_c(self, t: int) -> None:
        self.__set_dual('c', t, (
            self.__get_dual('c', t) + self.__get_rho('c') * (self.__get_gate('c', t) - (
                self.__get_gate('f', t) * self.__get_gate('c', t - 1)
                + self.__get_gate('i', t) * self.__get_gate('g', t)
            ))
        ))

    def __update_dual_h(self, t: int) -> None:
        if t < self.seq_len:
            return
        self.__set_dual('h', t, (
            self.__get_dual('h', t) + self.__get_rho('h') * (
                self.__get_gate('h', t) - self.__get_gate('o', t) * self.__tanh(self.__get_gate('c', t))
            )
        ))

    def __update_dual_y(self) -> None:
        self.__set_dual_y(
            self.__get_dual_y() + self.__get_rho('y') * (
                    self.__get_a() - self.__get_gate('h', self.seq_len) @ self.__get_wy()
            )
        )
