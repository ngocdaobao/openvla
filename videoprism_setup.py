import csv

from openvla.videoprism import models as vp
import jax
import jax.numpy as jnp
try:
	from flax.linen import summary as flax_summary
except Exception:
	flax_summary = None

model_name = 'videoprism_public_v1_base'
flax_model = vp.get_model(model_name)

# Print and save architecture summary using a dummy input.
dummy_video = jnp.zeros((1, 16, 288, 288, 3), dtype=jnp.float32)
summary = None
if flax_summary is not None:
	try:
		summary = flax_model.tabulate(
				jax.random.PRNGKey(0),
				dummy_video,
				train=False,
				depth=None,
				return_summary=True,
		)
	except TypeError:
		summary = None

if summary is None:
	arch = flax_model.tabulate(jax.random.PRNGKey(0), dummy_video, train=False)
	print(arch)
	with open("videoprism_arch.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["line"])
		for line in arch.splitlines():
			writer.writerow([line])
else:
	rows = getattr(summary, "rows", [])
	with open("videoprism_arch.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["path", "module_type", "param_shapes"])
		for row in rows:
			path = getattr(row, "path", "") or ""
			module_obj = getattr(row, "module", None)
			module_type = module_obj.__class__.__name__ if module_obj is not None else "Module"
			params = getattr(row, "params", None)
			shapes = []
			if hasattr(params, "items"):
				for key, value in params.items():
					if hasattr(value, "shape"):
						shapes.append(f"{key}={tuple(value.shape)}")
			writer.writerow([path, module_type, "; ".join(shapes)])

	print(f"Saved CSV to videoprism_arch.csv with {len(rows)} rows.")
