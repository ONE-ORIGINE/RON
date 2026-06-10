from __future__ import annotations
from ron.datasets.synthetic_dynamics import make_dynamics_batch
from ron.datasets.collision_world import CollisionWorldConfig, make_collision_batch
from ron.networks import RONPureDynamicsNet, RONTriaxialTeacherNet, RONHybridTriFamilyNet

KW = dict(device="cpu", lite_cells=4, memory_slots=8, grid=1, cube=3, moment_degree=1, signature_degree=1, harmonic_degree=1, internal_cycles=1)

def test_synthetic_and_collision_shapes():
    dyn = make_dynamics_batch(3, seed=1)
    col = make_collision_batch(CollisionWorldConfig(batch_size=3, seed=2))
    assert dyn.axes.shape == (3,3,3)
    assert dyn.energy.shape == (3,3)
    assert col.axes.shape == (3,3,3)
    assert col.energy.shape == (3,3)

def test_pure_triaxial_hybrid_networks():
    dyn = make_dynamics_batch(3, seed=3)
    pure = RONPureDynamicsNet(**KW).infer_axes(dyn.axes, dyn.energy, dyn.mask)
    assert pure.summary["mode"] == "ron_pure"
    tri = RONTriaxialTeacherNet(**KW).infer_axes(dyn.axes, dyn.energy, dyn.mask)
    assert tri.summary["triaxial_teacher_mode_count"] == 6
    hyb = RONHybridTriFamilyNet(**KW).infer_states(dyn.x_state, dyn.y_state, dyn.z_state, dyn.mask)
    assert hyb.summary["hybrid_accepted_count"] == 3
    assert hyb.summary["hybrid_rejected_count"] == 0
