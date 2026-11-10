"""Single source for tool schemas; no filesystem paths or executable code in model arguments."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Structure(Args):
    structure_id: str


class Potential(Structure):
    potential_id: str = "emt_demo"


class Relax(Potential):
    fmax_eV_per_A: float = Field(default=0.03, gt=0, le=1)
    max_steps: int = Field(default=200, ge=1, le=2000)
    optimizer: Literal["FIRE", "BFGS"] = "FIRE"


class EOS(Potential):
    volume_factors: list[float] = Field(default_factory=lambda: [0.94,0.96,0.98,1.0,1.02,1.04,1.06], min_length=7, max_length=21)


class MD(Potential):
    ensemble: Literal["NVE", "NVT"] = "NVE"
    temperature_K: float = Field(default=100., gt=0, le=2000)
    timestep_fs: float = Field(default=0.5, gt=0, le=2)
    equilibration_steps: int = Field(default=20, ge=0, le=5000)
    production_steps: int = Field(default=100, ge=2, le=10000)
    sample_interval: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2**32-1)


class Job(Args):
    job_id: str


class Analysis(Args):
    trajectory_id: str
    element: str
    remove_com_drift: bool = False


class Empty(Args):
    pass


TOOLS = {
    "inspect_structure": (Structure, "Inspect a registered structure; returns composition, PBC, and geometry diagnostics."),
    "list_potentials": (Empty, "List administrator-registered potentials, hardware, domain limits, and demo status."),
    "check_potential": (Potential, "Check element coverage and registry restrictions. Compatibility is not an accuracy guarantee."),
    "evaluate_energy_forces": (Potential, "Compute total energy in eV and forces in eV/Angstrom; queues a job."),
    "submit_relaxation": (Relax, "Queue fixed-cell atomic relaxation; convergence requires final maximum force below requested tolerance."),
    "submit_eos": (EOS, "Queue fixed-fractional-coordinate volume scan and Birch-Murnaghan EOS fit. No internal relaxation per volume."),
    "submit_md": (MD, "Queue fixed-cell ASE MD. Equilibration uses Langevin, production is NVE or Langevin NVT. Output positions remain unwrapped."),
    "get_job_status": (Job, "Read job state; queued/running is not completed or converged."),
    "get_result": (Job, "Get completed calculation result and immutable evidence ID."),
    "cancel_job": (Job, "Request cancellation, checked cooperatively between calculation steps."),
    "analyze_trajectory": (Analysis, "Analyze species-resolved time-origin-averaged MSD. Returns provisional fits, never certifies converged diffusion."),
}


def tool_schemas():
    return [{"type":"function", "function":{"name":name,"description":description,
             "parameters":model.model_json_schema()}} for name,(model,description) in TOOLS.items()]

