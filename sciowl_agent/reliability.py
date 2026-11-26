"""Deterministic execution checks and scientific outcome reporting."""
import math

CALCULATIONS = frozenset({'evaluate_energy_forces', 'evaluate_structures',
                         'submit_relaxation', 'submit_eos', 'submit_md'})


def preflight(atoms, operation, arguments):
    """Cheap checks before queueing and again before calculator construction."""
    from .science import inspect
    inspection = inspect(atoms)
    if operation == 'submit_eos':
        if not atoms.pbc.all():
            raise ValueError('EOS requires fully periodic bulk structure')
        factors = arguments['volume_factors']
        if len(set(factors)) != len(factors) or any(not 0.8 <= x <= 1.2 for x in factors):
            raise ValueError('EOS factors must be unique and within [0.8,1.2]')
    if operation == 'submit_md' and arguments['production_steps'] // arguments['sample_interval'] + 1 < 3:
        raise ValueError('Need at least three sampled frames')
    if arguments.get('stress_policy') == 'required' and (
            not atoms.pbc.any() or inspection['volume_A3'] <= 1e-8):
        raise ValueError('Required stress needs a periodic nonsingular cell')
    return inspection


def check_constraints(constraints, name, arguments):
    if name not in CALCULATIONS:
        return
    potential = constraints.get('potential_id')
    if potential is not None and arguments['potential_id'] != potential:
        raise ValueError(f'Session requires potential_id {potential}; substitution is forbidden')
    allowed = constraints.get('allowed_operations')
    if allowed is not None and name not in allowed:
        raise ValueError(f'Operation {name} is excluded by session constraints')


def assess_job(store, job, evidence_ids):
    """Require each job's evidence and verify referenced artifacts before reporting.

    Acceptance refers only to the implemented numerical criterion, not accuracy.
    """
    status = job['status']
    operation = job['spec']['operation']
    if status == 'completed':
        rid = job['result']['result_id']
        if rid not in evidence_ids:
            raise ValueError('Final report must cite result evidence for every session job')
        result = store.get_json(rid, 'result')
        if result.get('operation') != operation:
            raise ValueError('Result operation does not match job specification')
        provenance = store.get_json(result['provenance_id'], 'provenance')
        if provenance.get('job_id') != job['id'] or provenance.get('spec') != job['spec']:
            raise ValueError('Result provenance does not match job specification')
        args = job['spec']['arguments']
        input_key = 'input_structure_ids' if operation == 'evaluate_structures' else 'input_structure_id'
        arg_key = 'structure_ids' if operation == 'evaluate_structures' else 'structure_id'
        if result.get(input_key) != args[arg_key] or result.get('potential_id') != args['potential_id']:
            raise ValueError('Result inputs do not match job specification')
        # Re-read through the integrity-checked artifact store, including provenance.
        for key in ('provenance_id', 'forces_id', 'predictions_id', 'trajectory_id', 'output_structure_id'):
            if result.get(key):
                store.artifact(result[key])
        outcome = 'calculated'
        criterion = None
        if operation == 'submit_relaxation':
            force = result['max_force_eV_per_A']
            threshold = job['spec']['arguments']['fmax_eV_per_A']
            if not math.isfinite(force):
                raise ValueError('Nonfinite final force')
            accepted = force < threshold
            if result.get('converged') is not accepted:
                raise ValueError('Convergence flag disagrees with force criterion')
            outcome = 'converged' if accepted else 'not_converged'
            criterion = {'max_force_eV_per_A': force, 'required_below_eV_per_A': threshold}
        elif operation == 'submit_eos':
            outcome = 'fit_checks_passed' if result.get('validated') is True else 'fit_checks_failed'
        elif operation == 'submit_md':
            outcome = 'provisional_trajectory'
        return {'job_id': job['id'], 'execution_status': status, 'scientific_outcome': outcome,
                'result_id': rid, 'criterion': criterion,
                'notice': 'Numerical checks do not establish model accuracy or validated diffusion.'}
    if status in ('queued', 'running'):
        raise ValueError('Cannot assess a pending job')
    for eid in evidence_ids:
        failure = store.get_json(eid)
        if failure.get('job_id') == job['id'] and failure.get('status') == status:
            return {'job_id': job['id'], 'execution_status': status,
                    'scientific_outcome': 'no_successful_result', 'result_id': eid}
    raise ValueError('Final report must cite failure evidence for every unsuccessful job')
