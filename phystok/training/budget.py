"""Explicit opportunity accounting; no TEST metric accepted by this API."""
from dataclasses import asdict, dataclass
import math
from phystok.utils.common import require
from .config import validate_budget


@dataclass
class Counters:
    presentations: int = 0
    attempted_updates: int = 0
    optimizer_updates: int = 0
    skipped_updates: int = 0
    nonfinite_batches: int = 0

    def record(self, n, *, updated, nonfinite=False):
        require(n > 0 and not (updated and nonfinite), 'invalid update counters')
        self.presentations += int(n)
        self.attempted_updates += 1
        self.optimizer_updates += int(updated)
        self.skipped_updates += int(not updated)
        self.nonfinite_batches += int(nonfinite)


class BudgetController:
    def __init__(self, policy, n_fit):
        self.policy = dict(validate_budget(policy))
        require(n_fit > 0, 'empty FIT')
        self.n_fit = int(n_fit)
        self.count = Counters()
        self.target_updates = policy['initial_updates']
        self.target_passes = policy['initial_passes']
        self.history = []
        self.best = math.inf
        self.best_index = -1
        self.bad_checks = 0
        self.lr_reductions = 0
        self.last_lr_reduction_index = -1
        self.extensions = []
        self.stop_reason = None

    @property
    def passes(self):
        return self.count.presentations / self.n_fit

    def hard_limit(self):
        p, c = self.policy, self.count
        return c.optimizer_updates >= p['max_updates'] or self.passes >= p['max_passes']

    def stage_limit(self):
        return self.count.optimizer_updates >= self.target_updates and self.passes >= self.target_passes

    def needs_validation(self):
        last = self.history[-1] if self.history else {'optimizer_updates':0,'presentations':0}
        return self.hard_limit() or self.stage_limit() or (
            self.count.optimizer_updates-last['optimizer_updates'] >= self.policy['val_every_updates'] and
            self.count.presentations-last['presentations'] >= self.policy['val_every_passes']*self.n_fit)

    def failure_reason(self):
        if self.count.nonfinite_batches > self.policy['max_nonfinite_batches']:
            return 'NONFINITE_LIMIT'
        if self.count.skipped_updates > self.policy['max_skipped_updates']:
            return 'SKIPPED_UPDATE_LIMIT'
        return None

    def validation(self, rmse, *, lr_before, lr_after, role='VAL'):
        require(role == 'VAL' and math.isfinite(rmse) and rmse >= 0, 'only finite legal VAL criterion allowed')
        i = len(self.history)
        improved = rmse < self.best * (1-self.policy['relative_min_delta'])
        exact_best = rmse < self.best
        if improved:
            self.bad_checks = 0
            self.best_index = i
        else:
            self.bad_checks += 1
        if exact_best:
            self.best = rmse
        if lr_after < lr_before:
            self.lr_reductions += 1
            self.last_lr_reduction_index = i
        recent = i-self.best_index < self.policy['recent_improvement_checks']
        qualified = (self.count.optimizer_updates >= self.policy['min_updates'] and self.passes >= self.policy['min_passes'])
        recovered = self.lr_reductions >= self.policy['required_lr_reductions'] and i-self.last_lr_reduction_index >= self.policy['recovery_checks']
        if self.hard_limit():
            self.stop_reason = 'RESOURCE_CAP_IMPROVING' if recent else 'RESOURCE_CAP_PLATEAU_UNCERTIFIED'
        elif qualified and recovered and self.bad_checks >= self.policy['patience_checks']:
            self.stop_reason = 'QUALIFIED_VAL_PLATEAU'
        elif self.stage_limit():
            # Extend for recent improvement or if recovery qualification is not
            # yet satisfied. No hidden per-model discretionary extension.
            old = [self.target_updates,self.target_passes]
            self.target_updates = min(self.policy['max_updates'], math.ceil(self.target_updates*self.policy['extension_factor']))
            self.target_passes = min(self.policy['max_passes'], self.target_passes*self.policy['extension_factor'])
            self.extensions.append({'at_check':i,'old':old,'new':[self.target_updates,self.target_passes],
                                    'reason':'RECENT_VAL_IMPROVEMENT' if recent else 'RECOVERY_NOT_QUALIFIED'})
        event = dict(asdict(self.count), passes=self.passes, validation_index=i, val_rmse=rmse,
                     learning_rate_before=lr_before,learning_rate_after=lr_after,significant_improvement=improved,
                     exact_best=exact_best,best_val_rmse=self.best,lr_reductions=self.lr_reductions,
                     minimum_opportunity_reached=qualified,at_cap_improving=bool(self.hard_limit() and recent),
                     resource_limit_reached=self.hard_limit(),stop_reason=self.stop_reason)
        self.history.append(event)
        return event

    def state_dict(self):
        d = dict(self.__dict__)
        d['count'] = asdict(self.count)
        return d

    @classmethod
    def from_state(cls, state):
        obj = cls(state['policy'],state['n_fit'])
        obj.__dict__.update(state)
        obj.count = Counters(**state['count'])
        return obj
