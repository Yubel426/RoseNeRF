import jax.numpy as jnp

def get_rays_uvst(rays_o, rays_d, z1=0, z2=1):
    # rays_o[batch_size, 3], rays_d[batch_size, 3]
    x0, y0, z0 = jnp.split(rays_o, 3, axis=-1)
    a, b, c = jnp.split(rays_d, 3, axis=-1)
    # 计算交点
    t1 = (z1 - z0) / c
    t2 = (z2 - z0) / c
    x1 = x0 + a * t1
    x2 = x0 + a * t2
    y1 = y0 + b * t1
    y2 = y0 + b * t2
    uvst = jnp.concatenate([x1, y1, x2, y2], axis=-1)
    return uvst

def get_interest_point(uvst, z_val):
    # uvst[batch_size, 4]
    # z_val[batch_size, 1]
    z_val = jnp.full((*uvst.shape[:-1],1), z_val)
    x1, y1, x2, y2 = jnp.split(uvst, 4, axis=-1)
    # 计算交点
    # z1 = 0, z2 = 1
    x0 = (x2 - x1) * z_val + x1
    y0 = (y2 - y1) * z_val + y1
    interest_point = jnp.concatenate([x0, y0, z_val], axis=-1)
    return interest_point

def normalize(v):
    """Normalize a vector."""
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

def get_rays_d(uvst, z1=0, z2=1):
    # uvst[batch_size, 4]
    x1, y1, x2, y2 = jnp.split(uvst, 4, axis=-1)
    # 计算光线方向
    a = x2 - x1
    b = y2 - y1
    c = jnp.full_like(a, z2 - z1)  # Assuming z1=0 and z2=1
    # 方向归一化
    rays_d = normalize(jnp.concatenate([a, b, c], axis=-1))
    return rays_d
