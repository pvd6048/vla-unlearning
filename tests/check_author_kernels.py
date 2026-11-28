"""Optional CPU comparison with pinned author sources; see GU_SHARP_AUDIT.md for downloads.

This checks isolated update kernels, not equivalence of the complete unlearning algorithms.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
import copy

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.gu import retain_null
from vlu.sharp_minmax import SharpMinMaxUpdate


def checked_source(path, expected_hash):
    source = path.read_text(encoding="utf-8")  # normalize CRLF before hashing
    # and the final newline (raw GitHub's sam.py has none; an editor adds one)
    if hashlib.sha256((source.rstrip("\n") + "\n").encode()).hexdigest() != expected_hash:
        raise ValueError(f"{path}: source differs from the inspected author revision")
    return source


class Nonlinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.7, -1.2, .4], dtype=torch.float64))

    def forward(self, inputs):
        x = self.weight  # coupled entries: the perturbation of one partition changes the other's gradient
        return {"loss": (x[0] + 2*x[1] - x[2]).square() + .2*x.pow(4).sum()}


def check_sharp(source):
    namespace = {}
    exec(compile(source, "author_sam.py", "exec"), namespace)
    cases, max_error = 0, 0.
    for forget in (False, True):
        for strength in (0., .5, 1., 2.):
            for adaptive in (False, True):
                for rho in (0., .13):
                    own, reference = Nonlinear(), Nonlinear()
                    mask = torch.tensor([True, False, True])
                    active = mask if forget else ~mask
                    inputs = {}
                    coefficient = -.3 if forget else .7
                    rule = SharpMinMaxUpdate({"weight": mask}, .7, rho, strength, adaptive)
                    optimizer = torch.optim.SGD(own.parameters(), lr=.02, momentum=.9, weight_decay=5e-4)
                    other = namespace["SAM"](reference.parameters(), torch.optim.SGD,
                                             sign="max" if forget else "min", rho=rho, lamb=strength,
                                             adaptive=adaptive, lr=.02, momentum=.9, weight_decay=5e-4)
                    for _ in range(4):
                        # Our literal blend, applied by the unmodified native base optimizer.
                        _, grads, _ = rule._phase(own, inputs, {"weight": own.weight}, forget)
                        own.weight.grad = grads[0]
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # The authors' Algorithm 2 phase: first step on the full gradient, second backward,
                        # p.grad *= active (their partition mask), second step.
                        (coefficient * reference(inputs)["loss"]).backward()
                        other.first_step(zero_grad=True)
                        (coefficient * reference(inputs)["loss"]).backward()
                        reference.weight.grad.mul_(active)
                        other.second_step(zero_grad=True)
                        max_error = max(max_error, (own.weight-reference.weight).abs().max().item())
                        torch.testing.assert_close(own.weight, reference.weight, atol=1e-12, rtol=1e-12)
                    cases += 1
    print(f"Sharp author SAM: {cases} configurations x 4 momentum/decay steps; max abs error {max_error:.3g}")


def check_gu(source):
    # Extract only the projector class; the full trainer needs unrelated LLM dependencies.
    tree = ast.parse(source)
    projector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetainNullProjector")
    namespace = {"torch": torch, "re": re}
    exec(compile(ast.Module(body=[projector], type_ignores=[]), "author_gu.py", "exec"), namespace)
    reference = namespace["RetainNullProjector"](nn.Linear(9, 1), ["weight"], use_adam_diag=False)
    max_error = 0.
    for seed in range(20):
        torch.manual_seed(seed)
        q = torch.linalg.qr(torch.randn(9, 3)).Q
        d = torch.randn(9)
        expected = reference._project_tensor(d.clone(), list(q.T), mode="perp")
        actual = retain_null(d, list(q.T), torch.ones_like(d))
        max_error = max(max_error, (expected-actual).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    print(f"GU author Euclidean projection primitive: 20 seeded cases; max abs error {max_error:.3g}")


def check_gu_npo_rt(source):
    """vlu/gu_npo_rt.py against the authors' RetainNullProjector with Adam whitening bound to the optimizer:
    the same float16 basis from the same retain gradients (Gram-Schmidt, residual threshold, size cap), and
    the same whitened perpendicular / tangent projections mapped back."""
    from vlu.gu_npo_rt import GuNpoRt
    tree = ast.parse(source)
    projector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetainNullProjector")
    namespace = {"torch": torch, "re": re}
    exec(compile(ast.Module(body=[projector], type_ignores=[]), "author_gu.py", "exec"), namespace)
    max_error = 0.
    for seed in range(10):
        torch.manual_seed(seed)
        model = nn.Linear(12, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 12)).square().sum().backward()
        optimizer.step()  # gives exp_avg_sq
        optimizer.zero_grad()
        reference = namespace["RetainNullProjector"](model, ["weight"], k=3, use_adam_diag=True, use_opt_state=True,
                                                     basis_dtype=torch.float16, basis_update_every=1,
                                                     residual_keep_thresh=1e-3)
        reference.bind_optimizer(optimizer)
        rule = GuNpoRt(.1, ["weight"], 3, 1, .5, gamma=1., alpha=1.)
        rule.initialize(optimizer)
        xs = [torch.randn(3, 12) for _ in range(4)]
        xs.insert(2, 2 * xs[0])  # a parallel gradient: kept by neither
        for x in xs:
            loss = model(x).square().sum()
            grad = torch.autograd.grad(loss, [model.weight], retain_graph=True)[0]
            reference.maybe_update_basis_with_retain(loss)
            rule._grow_basis({"weight": model.weight}, [grad])
        ours, theirs = rule.basis["weight"], reference.basis["weight"]
        assert len(ours) == len(theirs) == 3, (len(ours), len(theirs))
        for u, w in zip(ours, theirs):
            max_error = max(max_error, (u.float() - w.float()).abs().max().item())
            torch.testing.assert_close(u, w, atol=2e-3, rtol=0)
        g_f, g_r = torch.randn(1, 12), torch.randn(1, 12)
        rule.basis["weight"] = list(theirs)  # the same float16 vectors for the projection check
        whiten = lambda g: reference._precond("weight", g)
        back = lambda g: reference._deprecond("weight", g)
        perp = back(reference._project_tensor(whiten(g_f).float(), theirs, mode="perp"))
        tan = back(reference._project_tensor(whiten(g_r).float(), theirs, mode="tan"))
        torch.testing.assert_close(rule._project("weight", model.weight, g_f.clone(), None), perp, atol=1e-6, rtol=1e-5)
        own_r = rule._project("weight", model.weight, None, g_r.clone())
        torch.testing.assert_close(own_r, tan, atol=1e-6, rtol=1e-5)
        max_error = max(max_error, (own_r - tan).abs().max().item())
    print(f"GU on NPO-RT vs the authors' projector (Adam metric): 10 seeds, basis and projections; max abs error {max_error:.3g}")


def check_gu_algorithm_1(source):
    """Execute the author's optimizer_step unchanged, with a tiny classifier and a no-op parent step.
    This includes recovery, sign gating, cap, both weights and the full Adam-metric projection.
    Only the LLM trainer scaffolding is replaced; its KL is evaluated literally.
    """
    from vlu.gu_npo_rt import GuNpoRt
    tree = ast.parse(source)
    projector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetainNullProjector")
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GeometricUnlearn")
    trainer.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    trainer.body = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and
                    n.name in ("optimizer_step", "nullProjProjectTan")]
    class Parent:
        def optimizer_step(self, *args, **kwargs): pass
    namespace = {"torch": torch, "re": re, "F": torch.nn.functional, "Parent": Parent}
    module = ast.fix_missing_locations(ast.Module(body=[projector, trainer], type_ignores=[]))
    exec(compile(module, "author_gu_algorithm_1.py", "exec"), namespace)
    class Classifier(nn.Linear):
        def forward(self, x):
            return SimpleNamespace(logits=super().forward(x))

    error = 0.
    for seed in range(10):
        torch.manual_seed(seed)
        model = Classifier(4, 3, bias=False)
        ref = copy.deepcopy(model).requires_grad_(False)
        with torch.no_grad(): model.weight.add_(.2 * torch.randn_like(model.weight))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.state[model.weight] = {"exp_avg_sq": torch.rand_like(model.weight) + .1}
        projection = namespace["RetainNullProjector"](model, ["weight"], k=2, use_adam_diag=True,
                                                      use_opt_state=True, basis_dtype=torch.float16)
        projection.bind_optimizer(optimizer)
        # The released trainer differentiates the anchor twice on refresh steps without retaining
        # its graph. Compare its unmodified non-refresh rewrite; basis growth is checked separately.
        projection.basis_update_every, projection._step = 2, 1
        q = torch.linalg.qr(torch.randn(12, 2)).Q
        projection.basis["weight"] = [u.reshape_as(model.weight).half() for u in q.T]
        author = namespace["GeometricUnlearn"]()
        author.model, author.ref_model, author.null_proj = model, ref, projection
        author._last_retain_inputs = {"x": torch.randn(5, 4)}
        author._last_forget_inputs = {"x": torch.randn(5, 4)}
        author.gamma, author.alpha = 1.5, .7
        author.sign_selective, author.sign_tau, author.sign_cap_ratio = True, 0., .5
        x = author._last_retain_inputs
        anchor = torch.nn.functional.kl_div(model(**x).logits.log_softmax(-1),
                                            ref(**x).logits.softmax(-1), reduction="batchmean")
        gr = torch.autograd.grad(anchor, model.weight)[0]
        total = torch.randn_like(model.weight)
        own = GuNpoRt(.1, ["weight"], 2, 2, .5, gamma=author.gamma, alpha=author.alpha)
        own.initialize(optimizer)
        own.basis["weight"] = list(projection.basis["weight"])
        recovered = (total - author.alpha * gr) / author.gamma
        actual = own._project("weight", model.weight, recovered, gr)
        model.weight.grad = total.clone()
        author.optimizer_step(optimizer=optimizer)
        error = max(error, (actual-model.weight.grad).abs().max().item())
        torch.testing.assert_close(actual, model.weight.grad, atol=1e-6, rtol=1e-5)
        # gu_recovery="separate": with the anchor as the retain loss inside the total (Algorithm 1's own
        # setting), the authors' recovery yields exactly the forget gradient, so their rewrite must equal ours
        # given the two loss gradients separately.
        g_forget = torch.randn_like(model.weight)
        separate = GuNpoRt(.1, ["weight"], 2, 2, .5, gamma=author.gamma, alpha=author.alpha, recovery="separate")
        separate.initialize(optimizer)
        separate.basis["weight"] = list(projection.basis["weight"])
        actual = separate._project("weight", model.weight, g_forget, gr)
        model.weight.grad = author.gamma * g_forget + author.alpha * gr
        projection._step = 1  # again a non-refresh step (see above); the basis is unchanged
        author.optimizer_step(optimizer=optimizer)
        error = max(error, (actual-model.weight.grad).abs().max().item())
        torch.testing.assert_close(actual, model.weight.grad, atol=1e-6, rtol=1e-5)
    print(f"GU complete author gradient rewrite: 10 seeds, gamma=1.5, alpha=.7, both recoveries; max abs error {error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gu_source", type=Path)
    parser.add_argument("sharp_source", type=Path)
    args = parser.parse_args()
    gu = checked_source(args.gu_source, "e9878ec852083bb86704f320eaf5882da872b7d17b28a4e054dc6d6152cb5c55")
    sharp = checked_source(args.sharp_source, "437de1860ddcf4d8c0db08696dfc576ac2e29f5f45bbd0f82d8568e13af381ae")
    torch.set_num_threads(1)
    check_sharp(sharp)
    check_gu(gu)
    check_gu_npo_rt(gu)
    check_gu_algorithm_1(gu)
    print("Kernel checks only; complete GR00T/LIBERO experiments are not reproduced. "
          "The primary GU method is npo_rt_gu; gu remains a legacy experimental method.")
