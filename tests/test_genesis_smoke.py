"""Sanity-check: Genesis can load Go2 URDF and step physics on GPU."""
import pytest
import torch


def test_genesis_loads_and_steps_go2():
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02),
        viewer_options=gs.options.ViewerOptions(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.0, 0.0, 0.42),
        ),
    )
    scene.build(n_envs=2)

    for _ in range(10):
        scene.step()

    pos = robot.get_pos()
    assert pos.shape == (2, 3)
    assert torch.isfinite(pos).all()
