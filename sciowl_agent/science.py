"""Real ASE calculations. EMT is a classical demo potential, not an MLIP."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase import Atoms, units
from ase.io import read, write, iread
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


def import_structures(store, path):
    """Import all frames in order; duplicates keep their source frame identity."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ids = []
    for index, atoms in enumerate(iread(path, index=":")):
        if index >= 1000:
            raise ValueError("At most 1000 frames per import; split the input file")
        inspect(atoms)
        ids.append(atoms_to_id(store, atoms, {"source_sha256": digest, "frame_index": index}))
    if not ids:
        raise ValueError("Input contains no structures")
    return ids


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

    def checked(self, atoms, pid, *, verify_assets=True):
        if pid not in self.entries:
            raise ValueError("Unregistered potential")
        cfg = dict(self.entries[pid])
        if not set(atoms.get_chemical_symbols()).issubset(cfg["elements"]):
            raise ValueError("Potential does not cover these elements")
        if len(atoms)>cfg.get("max_atoms",256):
            raise ValueError("Potential atom budget exceeded")
        if not verify_assets:
            return cfg
        if cfg["backend"] == "mace":
            if cfg.get("default_dtype", "float64") not in ("float32", "float64"):
                raise ValueError("MACE default_dtype must be float32 or float64")
            path = Path(cfg["model_path"]).resolve()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
                raise ValueError("MACE weight missing or SHA256 mismatch")
            if cfg.get("device","cpu").startswith("npu"):
                raise ValueError("Upstream MACE NPU execution is not validated by this release; use a separately validated backend")
            cfg["model_path"] = str(path)
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
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError("MACE backend requires mace-torch; install sciowl-agent[mlip] in the worker environment") from exc
    return MACECalculator(model_paths=cfg["model_path"], device=cfg.get("device","cpu"),
                          default_dtype=cfg.get("default_dtype", "float64"))


def finite_values(atoms):
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    if forces.shape != (len(atoms), 3) or not np.isfinite(energy) or not np.isfinite(forces).all():
        raise ValueError("Nonfinite calculator output")
    return energy, forces


def predict_frame(atoms, stress_policy="auto"):
    energy, forces = finite_values(atoms)
    out = {"formula": atoms.get_chemical_formula(), "energy_eV": energy,
           "forces_eV_per_A": forces.tolist(), "stress_eV_per_A3": None,
           "stress_available": False,
           "stress_convention": "ASE tension-positive, Voigt xx yy zz yz xz xy"}
    if stress_policy == "skip":
        out["stress_unavailable_reason"] = "Stress calculation skipped by request"
    elif not atoms.pbc.any() or abs(np.linalg.det(atoms.cell)) < 1e-8:
        out["stress_unavailable_reason"] = "Stress requires a periodic structure with a nonsingular cell"
    else:
        try:
            stress = np.asarray(atoms.get_stress(voigt=True))
        except PropertyNotImplementedError:
            out["stress_unavailable_reason"] = "Calculator does not implement stress"
        else:
            if stress.shape != (6,) or not np.isfinite(stress).all():
                raise ValueError("Invalid or nonfinite stress")
            out.update(stress_eV_per_A3=stress.tolist(), stress_available=True)
    if stress_policy == "required" and not out["stress_available"]:
        raise ValueError(out["stress_unavailable_reason"])
    return out


def execute_batch(store, job, tick, progress):
    spec = job["spec"]
    args, cfg = spec["arguments"], spec["potential"]
    registry = Registry()
    registry.entries = {args["potential_id"]: cfg}
    work = store.root / "work" / job["id"]
    work.mkdir(exist_ok=True)
    calc = None
    path = work / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as output:
        for index, sid in enumerate(args["structure_ids"]):
            tick()
            atoms = load_atoms(store, sid)
            registry.checked(atoms, args["potential_id"], verify_assets=index == 0)
            if calc is None:
                if progress is not None:
                    progress("loading calculator")
                calc = calculator(cfg, work / "calculator")
            atoms.calc = calc
            record = {"index": index, "structure_id": sid,
                      **predict_frame(atoms, args["stress_policy"])}
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            tick()
            if progress is not None:
                progress(f"frame={index + 1}/{len(args['structure_ids'])} formula={record['formula']} energy={record['energy_eV']:.8g} eV stress_available={record['stress_available']}")
    predictions_id = store.put_bytes(path.read_bytes(), "predictions", ".jsonl", {"job_id": job["id"]})
    return {"operation": spec["operation"], "input_structure_ids": args["structure_ids"],
            "potential_id": args["potential_id"], "frames": len(args["structure_ids"]),
            "predictions_id": predictions_id, "demo_only": cfg.get("demo_only", True),
            "scientific_scope": cfg["domain"], "stress_policy": args["stress_policy"],
            "stress_convention": "ASE tension-positive, Voigt xx yy zz yz xz xy"}


def execute(store, job, tick, *, progress=None):
    spec = job["spec"]
    if spec["operation"] == "evaluate_structures":
        return execute_batch(store, job, tick, progress)
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
        if progress is not None:
            progress("evaluating energy and forces")
        prediction = predict_frame(atoms, a.get("stress_policy", "auto"))
        e, f = prediction["energy_eV"], np.asarray(prediction["forces_eV_per_A"])
        force_id = store.put_json({"forces_eV_per_A":f.tolist()}, "forces", {"job_id":job["id"]})
        fm = float(np.linalg.norm(f, axis=1).max())
        result.update(energy_eV=e, max_force_eV_per_A=fm, forces_id=force_id)
        if progress is not None:
            progress(f"energy={e:.8g} eV max_force={fm:.8g} eV/Å")
        result.update({k: v for k, v in prediction.items() if k != "forces_eV_per_A"})
        record = {"index": 0, "structure_id": a["structure_id"], **prediction}
        result["predictions_id"] = store.put_bytes(
            (json.dumps(record, allow_nan=False) + "\n").encode(), "predictions", ".jsonl", {"job_id": job["id"]})
        if prediction["stress_available"]:
            stress = np.asarray(prediction["stress_eV_per_A3"])
            result.update(stress_voigt_eV_per_A3=stress.tolist(),stress_voigt_GPa=(stress/units.GPa).tolist(),
                          stress_convention="ASE tension-positive, Voigt xx yy zz yz xz xy")
    elif kind == "submit_relaxation":
        opt = {"FIRE":FIRE,"BFGS":BFGS}[a["optimizer"]](atoms, logfile=str(work/"optimizer.log"), trajectory=str(work/"relax.traj"))
        opt.attach(tick, interval=1)
        if progress is not None:
            progress(f"optimizer={a['optimizer']} target_max_force={a['fmax_eV_per_A']:.8g} eV/Å")

            def report_step():
                detail = f"step={opt.nsteps}/{a['max_steps']}"
                # Reuse the values ASE just calculated for its optimizer log.
                energy = atoms.calc.results.get("energy")
                forces = atoms.calc.results.get("forces")
                if energy is not None and forces is not None:
                    max_force = np.linalg.norm(forces, axis=1).max()
                    detail += f" energy={energy:.8g} eV max_force={max_force:.8g} eV/Å"
                progress(detail)

            opt.attach(report_step, interval=1)
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
        if progress is not None:
            progress(f"scanning {len(factors)} volumes")
        for index, factor in enumerate(factors, start=1):
            tick()
            atoms.set_cell(base*factor**(1/3),scale_atoms=True)
            inspect(atoms)
            e,_ = finite_values(atoms)
            volumes.append(float(atoms.get_volume())); energies.append(e)
            if progress is not None:
                progress(f"point={index}/{len(factors)} volume={volumes[-1]:.8g} Å³ energy={e:.8g} eV")
        if progress is not None:
            progress("fitting Birch–Murnaghan equation of state")
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
        if progress is not None:
            progress(f"equilibration steps={a['equilibration_steps']} target_temperature={a['temperature_K']:.8g} K")

            def report_equilibration():
                progress(f"equilibration step={equil.nsteps}/{a['equilibration_steps']} temperature={atoms.get_temperature():.8g} K")

            if a["equilibration_steps"]:
                equil.attach(report_equilibration, interval=a["sample_interval"])
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
            if progress is not None:
                progress(f"production step={dyn.nsteps}/{a['production_steps']} time={times[-1]:.8g} fs temperature={t:.8g} K total_energy={energies[-1]:.8g} eV")
        dyn.attach(tick,interval=1)
        dyn.attach(sample,interval=a["sample_interval"])
        if progress is not None:
            progress(f"production ensemble={a['ensemble']} steps={a['production_steps']}")
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

