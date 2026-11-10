"""Real ASE calculations. EMT is a classical demo potential, not an MLIP."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase import Atoms, units
from ase.io import read, write
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase.optimize import FIRE, BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.constraints import FixCom


DEFAULT_REGISTRY = {"emt_demo": {
    "backend":"emt", "elements":["Al","Cu","Ag","Au","Ni","Pd","Pt"],
    "device":"cpu", "demo_only":True,
    "domain":"ASE EMT classical potential demonstration; not validated for research predictions",
    "max_atoms":256, "revision":"ase-emt-v1"}}


def atoms_to_id(store, atoms, metadata=None):
    # JSON structures avoid deserializing model-supplied pickle/calculator objects.
    payload = {"numbers":atoms.numbers.tolist(), "positions":atoms.positions.tolist(),
               "cell":atoms.cell.array.tolist(), "pbc":atoms.pbc.tolist()}
    return store.put_json(payload, "structure", metadata)


def load_atoms(store, aid):
    p = store.get_json(aid, "structure")
    atoms = Atoms(numbers=p["numbers"], positions=p["positions"], cell=p["cell"], pbc=p["pbc"])
    inspect(atoms)
    return atoms


def import_structure(store, path):
    atoms = read(path, index=0)
    inspect(atoms)
    return atoms_to_id(store, atoms, {"source_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()})


def inspect(atoms):
    if not 1 <= len(atoms) <= 256:
        raise ValueError("Initial release supports 1..256 atoms")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("Nonfinite structure")
    if atoms.pbc.any() and abs(np.linalg.det(atoms.cell)) < 1e-8:
        raise ValueError("Periodic structure requires a nonsingular cell")
    distances = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min()) if len(atoms)>1 else None
    if minimum is not None and minimum < 0.5:
        raise ValueError("Atomic overlap: minimum distance below 0.5 Angstrom")
    return {"formula":atoms.get_chemical_formula(), "elements":sorted(set(atoms.get_chemical_symbols())),
            "n_atoms":len(atoms), "pbc":atoms.pbc.tolist(), "volume_A3":float(abs(np.linalg.det(atoms.cell))),
            "min_distance_A":minimum}


class Registry:
    def __init__(self, path=None):
        self.entries = json.loads(Path(path).read_text()) if path else DEFAULT_REGISTRY

    def checked(self, atoms, pid):
        if pid not in self.entries:
            raise ValueError("Unregistered potential")
        cfg = dict(self.entries[pid])
        if not set(atoms.get_chemical_symbols()).issubset(cfg["elements"]):
            raise ValueError("Potential does not cover these elements")
        if len(atoms)>cfg.get("max_atoms",256):
            raise ValueError("Potential atom budget exceeded")
        if cfg["backend"] == "mace":
            path = Path(cfg["model_path"]).resolve()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
                raise ValueError("MACE weight missing or SHA256 mismatch")
            if cfg.get("device","cpu").startswith("npu"):
                raise ValueError("Upstream MACE NPU execution is not validated by this release; use a separately validated backend")
        elif cfg["backend"] == "espresso":
            for name,digest in cfg["pseudopotential_sha256"].items():
                pp=(Path(cfg["pseudo_dir"])/name).resolve()
                if pp.parent != Path(cfg["pseudo_dir"]).resolve() or not pp.is_file() or hashlib.sha256(pp.read_bytes()).hexdigest()!=digest:
                    raise ValueError("Pseudopotential missing or hash mismatch")
            if not set(cfg["pseudopotentials"].values()).issubset(cfg["pseudopotential_sha256"]):
                raise ValueError("All pseudopotentials require hashes")
        elif cfg["backend"] != "emt":
            raise ValueError("Unsupported calculator backend")
        return cfg


def calculator(cfg,directory=None):
    if cfg["backend"] == "emt":
        return EMT()
    if cfg["backend"] == "espresso":
        from ase.calculators.espresso import Espresso,EspressoProfile
        if directory is None:
            raise ValueError("Espresso requires an isolated job directory")
        profile=EspressoProfile(command=cfg["command"],pseudo_dir=cfg["pseudo_dir"])
        return Espresso(profile=profile,pseudopotentials=cfg["pseudopotentials"],
                        input_data=cfg["input_data"],kpts=tuple(cfg["kpts"]),directory=str(directory))
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=cfg["model_path"], device=cfg.get("device","cpu"), default_dtype="float64")


def finite_values(atoms):
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("Nonfinite calculator output")
    return energy, forces


def execute(store, job, tick):
    spec = job["spec"]
    a = spec["arguments"]
    atoms = load_atoms(store, a["structure_id"])
    cfg = spec["potential"]
    # Revalidate snapshot at execution time, including weight integrity.
    RegistrySnapshot = Registry()
    RegistrySnapshot.entries = {a["potential_id"]:cfg}
    RegistrySnapshot.checked(atoms, a["potential_id"])
    before = inspect(atoms)
    work = store.root / "work" / job["id"]
    work.mkdir(exist_ok=True)
    atoms.calc = calculator(cfg,work/"calculator")
    tick()
    kind = spec["operation"]
    result = {"operation":kind, "input_structure_id":a["structure_id"], "potential_id":a["potential_id"],
              "demo_only":cfg.get("demo_only",True), "scientific_scope":cfg["domain"], "n_atoms":len(atoms)}
    if kind == "evaluate_energy_forces":
        e,f = finite_values(atoms)
        force_id = store.put_json({"forces_eV_per_A":f.tolist()}, "forces", {"job_id":job["id"]})
        result.update(energy_eV=e, max_force_eV_per_A=float(np.linalg.norm(f,axis=1).max()), forces_id=force_id)
        try:
            stress=atoms.get_stress(voigt=True)
            if not np.isfinite(stress).all():raise ValueError("Nonfinite stress")
            result.update(stress_voigt_eV_per_A3=stress.tolist(),stress_voigt_GPa=(stress/units.GPa).tolist(),
                          stress_convention="ASE tension-positive, Voigt xx yy zz yz xz xy")
        except PropertyNotImplementedError:
            result["stress_available"]=False
    elif kind == "submit_relaxation":
        opt = {"FIRE":FIRE,"BFGS":BFGS}[a["optimizer"]](atoms, logfile=str(work/"optimizer.log"), trajectory=str(work/"relax.traj"))
        opt.attach(tick, interval=1)
        opt.run(fmax=a["fmax_eV_per_A"], steps=a["max_steps"])
        e,f = finite_values(atoms)
        fm = float(np.linalg.norm(f,axis=1).max())
        final = atoms_to_id(store, atoms, {"job_id":job["id"]})
        result.update(energy_eV=e,max_force_eV_per_A=fm,converged=fm<a["fmax_eV_per_A"],
                      requested_fmax_eV_per_A=a["fmax_eV_per_A"], steps=opt.nsteps, output_structure_id=final,
                      cell_policy="fixed")
    elif kind == "submit_eos":
        if not atoms.pbc.all():
            raise ValueError("EOS requires fully periodic bulk structure")
        factors = sorted(a["volume_factors"])
        if len(set(factors)) != len(factors) or min(factors)<0.8 or max(factors)>1.2:
            raise ValueError("EOS factors must be unique and within [0.8,1.2]")
        base = atoms.cell.copy()
        volumes, energies = [], []
        for factor in factors:
            tick()
            atoms.set_cell(base*factor**(1/3),scale_atoms=True)
            inspect(atoms)
            e,_ = finite_values(atoms)
            volumes.append(float(atoms.get_volume())); energies.append(e)
        eos = EquationOfState(volumes,energies,eos="birchmurnaghan")
        v0,e0,b = eos.fit()
        predictions = eos.func(np.array(volumes),*eos.eos_parameters)
        valid = min(volumes)<v0<max(volumes) and b>0 and 0<int(np.argmin(energies))<len(energies)-1
        result.update(volumes_A3=volumes,energies_eV=energies, equilibrium_volume_A3=float(v0),
                      equilibrium_energy_eV=float(e0),bulk_modulus_GPa=float(b/units.GPa),
                      fit_rmse_eV=float(np.sqrt(np.mean((predictions-energies)**2))),
                      validated=bool(valid),protocol="fixed fractional coordinates; no per-volume relaxation")
    elif kind == "submit_md":
        rng = np.random.default_rng(a["seed"])
        MaxwellBoltzmannDistribution(atoms,temperature_K=a["temperature_K"],rng=rng)
        if len(atoms)>1:
            Stationary(atoms)
        if len(atoms)>1:
            atoms.set_constraint(FixCom())
        equil = Langevin(atoms,a["timestep_fs"]*units.fs,temperature_K=a["temperature_K"],friction=0.01/units.fs,rng=rng,fixcm=False)
        equil.attach(tick,interval=1)
        equil.run(a["equilibration_steps"])
        dyn = VelocityVerlet(atoms,a["timestep_fs"]*units.fs) if a["ensemble"]=="NVE" else Langevin(
            atoms,a["timestep_fs"]*units.fs,temperature_K=a["temperature_K"],friction=0.01/units.fs,rng=rng,fixcm=False)
        positions, cells, times, energies, temps = [],[],[],[],[]
        def sample():
            tick()
            e,_=finite_values(atoms)
            t=float(atoms.get_temperature())
            if not np.isfinite(t) or t>max(5000.,5*a["temperature_K"]):
                raise ValueError("Temperature stability guard triggered")
            positions.append(atoms.positions.copy());cells.append(atoms.cell.array.copy())
            times.append(dyn.nsteps*a["timestep_fs"])
            energies.append(e+float(atoms.get_kinetic_energy()));temps.append(t)
        dyn.attach(tick,interval=1)
        dyn.attach(sample,interval=a["sample_interval"])
        dyn.run(a["production_steps"])
        if len(times)<3:
            raise ValueError("Need at least three sampled frames")
        blob=io.BytesIO()
        np.savez_compressed(blob,positions_A=np.array(positions),cell_A=np.array(cells),time_fs=np.array(times),
                            total_energy_eV=np.array(energies),temperature_K=np.array(temps),
                            masses=atoms.get_masses(),symbols=np.array(atoms.get_chemical_symbols()))
        tid=store.put_bytes(blob.getvalue(),"trajectory",".npz",{"job_id":job["id"],"unwrapped":True,"fixed_cell":True})
        result.update(trajectory_id=tid,frames=len(times),duration_fs=float(times[-1]),ensemble=a["ensemble"],
            temperature_mean_K=float(np.mean(temps)),energy_range_eV_per_atom=float(np.ptp(energies)/len(atoms)),
            seed=a["seed"],output_structure_id=atoms_to_id(store,atoms,{"job_id":job["id"]}),
            validation="short stability run; not a converged transport prediction")
    else:
        raise ValueError("Unknown worker operation")
    tick()
    result["input_inspection"]=before
    return result


def analyze_msd(store, trajectory_id, element, remove_com_drift=False):
    with np.load(store.artifact(trajectory_id,"trajectory"),allow_pickle=False) as z:
        p=z["positions_A"].copy(); t=z["time_fs"];symbols=z["symbols"];masses=z["masses"]
        if not np.allclose(z["cell_A"],z["cell_A"][0]):
            raise ValueError("Variable-cell trajectory unsupported")
    if len(t)<6 or not np.allclose(np.diff(t),np.diff(t)[0]):
        raise ValueError("MSD requires >=6 evenly sampled frames")
    mask=symbols==element
    if not mask.any():
        raise ValueError("Requested element absent")
    if remove_com_drift:
        com=np.average(p,axis=1,weights=masses)
        p-= (com-com[0])[:,None,:]
    q=p[:,mask,:]
    lags=np.arange(1,len(t)//2+1)
    per_axis=np.array([np.mean((q[k:]-q[:-k])**2,axis=(0,1)) for k in lags])
    lag_times=lags*(t[1]-t[0])
    start=len(lags)//2
    slopes=np.array([np.polyfit(lag_times[start:],per_axis[start:,i],1)[0] for i in range(3)])
    # 1 A^2/fs = 0.1 cm^2/s; per-axis Einstein slope = 2 D_ii.
    out={"trajectory_id":trajectory_id,"element":element,"lag_time_fs":lag_times.tolist(),
         "msd_A2":per_axis.sum(axis=1).tolist(),"msd_axes_A2":per_axis.tolist(),
         "candidate_D_axes_cm2_per_s":(slopes*0.05).tolist(),
         "candidate_D_iso_cm2_per_s":float(slopes.sum()/6*0.1),
         "remove_com_drift":remove_com_drift,"validated_diffusion":False,
         "warnings":["No diffusive-regime certification or replica uncertainty; fits are provisional and can be negative."]}
    return out

