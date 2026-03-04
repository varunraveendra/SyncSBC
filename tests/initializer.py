
import math, asyncio
from omni.isaac.dynamic_control import _dynamic_control as dc

# ---- Your target prim paths (children under each robot) ----
paths  = [f"/World/hero_plus_{i:02d}/base_link" for i in range(1, 3)]
paths1 = [f"/World/hero_plus_{i:02d}/base_link" for i in range(3, 6)]

def grid_xy(i, n, cols=4, dx=0.38, dy=0.7, z=0.0, yaw_deg=50.0, center=True, origin=(0.0, 0.0)):
    rows = math.ceil(n / cols)
    r, c = divmod(i, cols)
    if center:
        x = (c - (cols - 1)/2) * dx + origin[0]
        y = (r - (rows - 1)/2) * dy + origin[1]
    else:
        x = c * dx + origin[0]
        y = r * dy + origin[1]
    return (x, y, z), yaw_deg

PLACEMENTS1 = {p: grid_xy(i, n=len(paths),  cols=4, dx=0.38, dy=0.7, z=0.0, yaw_deg=50.0, center=True, origin=(0.0, 0.0))
               for i, p in enumerate(paths)}
PLACEMENTS2 = {p: grid_xy(i, n=len(paths1), cols=4, dx=0.38, dy=0.7, z=0.0, yaw_deg=20.0, center=True, origin=(0.0, 0.38))
               for i, p in enumerate(paths1)}

PLACEMENTS = {**PLACEMENTS1, **PLACEMENTS2}

# ---- Helpers ----
def yaw_deg_to_wxyz(yaw_deg: float):
    h = math.radians(yaw_deg) * 0.5
    return (math.cos(h), 0.0, 0.0, math.sin(h))  # (w,x,y,z)

def ancestors(path_str: str):
    parts = path_str.strip("/").split("/")
    for i in range(len(parts), 0, -1):
        yield "/" + "/".join(parts[:i])

def guess_robot_root(path_str: str):
    # "/World/hero_plus_03/base_link" -> "/World/hero_plus_03"
    parts = path_str.strip("/").split("/")
    return "/" + "/".join(parts[:2]) if len(parts) >= 2 else "/" + parts[0]

async def place_all(placements=PLACEMENTS, settle_frames: int = 2):
    """
    Assumes stage is already open and (ideally) playing.
    Does NOT start/stop the timeline. Only places robots.
    """
    app = None
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
    except Exception:
        pass

    # optional: let things settle a couple frames if app exists
    if app and settle_frames > 0:
        for _ in range(settle_frames):
            await app.next_update_async()

    dci = dc.acquire_dynamic_control_interface()

    for original_path, (pos, yaw_deg) in placements.items():
        art = dci.get_articulation(original_path)
        chosen_path = original_path

        if not art:
            for cand in ancestors(original_path):
                art = dci.get_articulation(cand)
                if art:
                    chosen_path = cand
                    break

        if not art:
            root_guess = guess_robot_root(original_path)
            art = dci.get_articulation(root_guess)
            if art:
                chosen_path = root_guess

        w, x, y, z = yaw_deg_to_wxyz(yaw_deg)
        T = dc.Transform()
        try:
            T.p = dc.Float3(*pos)
            T.r = dc.Float4(x, y, z, w)  # DC expects (x,y,z,w)
        except AttributeError:
            T.p = (pos[0], pos[1], pos[2])
            T.r = (x, y, z, w)

        if art:
            root = dci.get_articulation_root_body(art)
            dci.set_rigid_body_pose(root, T)
            dci.set_rigid_body_linear_velocity(root, (0.0, 0.0, 0.0))
            dci.set_rigid_body_angular_velocity(root, (0.0, 0.0, 0.0))
            dci.wake_up_articulation(art)
            print(f"[place] articulation placed @ {chosen_path} (requested {original_path})")
        else:
            rb = dci.get_rigid_body(original_path)
            if rb:
                dci.set_rigid_body_pose(rb, T)
                dci.set_rigid_body_linear_velocity(rb, (0.0, 0.0, 0.0))
                dci.set_rigid_body_angular_velocity(rb, (0.0, 0.0, 0.0))
                dci.wake_up_rigid_body(rb)
                print(f"[place] rigid body placed: {original_path}")
            else:
                print(f"[place] NOT FOUND (articulation or rigid): {original_path}")

    print("[place] done.")

# Kick it off (non-blocking) if you're in Script Editor:
import asyncio
asyncio.ensure_future(place_all())
